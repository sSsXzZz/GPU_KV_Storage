#ifndef _HASH_H_
#define _HASH_H_

#ifdef __clang__
// Added for YouCompleteMe
#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <host_defines.h>
int cudaConfigureCall(dim3 grid_size, dim3 block_size, unsigned shared_size = 0, cudaStream_t stream = 0);
#endif
#include <math.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>

// Constants for hash table
static constexpr uint64_t NUM_ELEMENTS = 1 << 20;  // 1M elements
static constexpr uint KEY_SIZE = 32;
static constexpr uint WORD_SIZE = 64;
static constexpr uint BATCH_SIZE = 10000;
static constexpr uint CPU_BATCH_SIZE = 10;

// Constants used for kernel invocation
static constexpr uint BLOCK_SIZE = 256;
static constexpr uint NUM_BLOCKS = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Constants for hash function
static constexpr uint32_t PRIME = 0x01000193;  //   16777619
static const uint32_t SEED = 0x811C9DC5;       // 2166136261

// Stores Hash table entry fields needed internally
class HashEntryInternal {
  public:
    bool occupied;
    char key[KEY_SIZE];
    char word[WORD_SIZE];
};

// ----------------------------------------------
// Cuda Memory
// ----------------------------------------------
class CudaManagedMemory {
  public:
    void* operator new(size_t len) {
        void* ptr;
        cudaMallocManaged(&ptr, len);
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaFree(ptr);
    }
};

class CudaMemory {
  public:
    void* operator new(size_t len) {
        void* ptr;
        cudaMalloc(&ptr, len);
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaFree(ptr);
    }
};

// ----------------------------------------------
// GPU Hash Table
// ----------------------------------------------
//

class GpuHashEntry : public CudaManagedMemory {
  public:
    GpuHashEntry() {
        std::memset(key, 0, sizeof(key));
        std::memset(word, 0, sizeof(word));
    }

    GpuHashEntry(char* k) : GpuHashEntry() {
        std::strcpy(key, k);
    }

    GpuHashEntry(char* k, char* w) : GpuHashEntry(k) {
        std::strcpy(word, w);
    }

    GpuHashEntry(GpuHashEntry& h) = delete;
    GpuHashEntry(GpuHashEntry&& h) = delete;

    void set(char* k) {
        std::strcpy(key, k);
    }

    void set(char* k, char* w) {
        std::strcpy(key, k);
        std::strcpy(word, w);
    }

    char key[KEY_SIZE];
    char word[WORD_SIZE];
};
inline bool operator==(const GpuHashEntry& a, const GpuHashEntry& b) {
    return (std::memcmp(a.key, b.key, KEY_SIZE) == 0) && (std::memcmp(a.word, b.word, WORD_SIZE) == 0);
}
inline std::ostream& operator<<(std::ostream& outs, const GpuHashEntry& entry) {
    return outs << "(" << entry.key << ", " << entry.word << ")";
}

class GpuHashEntryBatch : public CudaManagedMemory {
  public:
    GpuHashEntryBatch() {
        memset(entries, 0, sizeof(entries));
    }
    GpuHashEntryBatch(GpuHashEntryBatch& h) = delete;
    GpuHashEntryBatch(GpuHashEntryBatch&& h) = delete;

    GpuHashEntry entries[BATCH_SIZE];

    // Returns the entry at the given index
    inline GpuHashEntry& at(std::size_t i) {
        return entries[i];
    }
};

class GpuHashTable : public CudaMemory {
  public:
    // Inserts the entry into the hash table
    __device__ void insert_entry(GpuHashEntry* user_entry);

    // Finds the entry in the hash table. The word in the user_entry is set to the word found.
    __device__ void find_entry(GpuHashEntry* user_entry);

    // Returns the pointer to the entry at the given index
    __device__ HashEntryInternal* get_entry(uint32_t index);

    HashEntryInternal entries[NUM_ELEMENTS];

  private:
};

// ----------------------------------------------
// CPU Hash Table
// ----------------------------------------------

class CpuHashEntry {
  public:
    char key[KEY_SIZE];
    char word[WORD_SIZE];
};

class CpuHashEntryBatch {
  public:
    CpuHashEntry entries[CPU_BATCH_SIZE];
};

class CpuHashTable {
  public:
    // Insert entry into hash table
    void insert_entry(CpuHashEntry* user_entry);

    // Finds the entry in the hash table
    void find_entry(CpuHashEntry* user_entry);

    void insert_batch(CpuHashEntryBatch* entry_batch, uint num_entries);

    void find_batch(CpuHashEntryBatch* entry_batch, uint num_entries);

    HashEntryInternal entries[NUM_ELEMENTS];
};

// ----------------------------------------------
// GPU Hash Table Interface
// ----------------------------------------------
// These functions internally make kernel calls and synchronize afterwards.
//

void init_hash_table(GpuHashTable* hash_table);

void hash_insert(GpuHashTable* hash_table, GpuHashEntry* entry);

void hash_find(GpuHashTable* hash_table, GpuHashEntry* entry);

void hash_insert_batch(GpuHashTable* hash_table, GpuHashEntryBatch* entry_batch, uint num_entries);

void hash_find_batch(GpuHashTable* hash_table, GpuHashEntryBatch* entry_batch, uint num_entries);

// ----------------------------------------------
// Debugging Stuff
// ----------------------------------------------
void print_all_entries(GpuHashTable* hash_table);

#include <execinfo.h>
inline void print_trace(void) {
    void* trace[16];
    char** messages = (char**)NULL;
    int i, trace_size = 0;

    trace_size = backtrace(trace, 16);
    messages = backtrace_symbols(trace, trace_size);
    char addr[32];
    char filename[32];
    for (i = 1; i < trace_size; ++i) {
        printf("[bt] #%d %s\n", i, messages[i]);

        /* find first occurence of '(' or ' ' in message[i] and assume
         * everything before that is the file name. (Don't go beyond 0 though
         * (string terminator)*/
        size_t p = 0;
        while (messages[i][p] != '(' && messages[i][p] != ' ' && messages[i][p] != 0) ++p;
        std::memcpy(filename, messages[i], p);
        filename[p] = '\0';

        p++;
        size_t addr_start = p;
        while (messages[i][p] != ')' && messages[i][p] != ' ' && messages[i][p] != 0) ++p;
        std::memcpy(addr, messages[i] + addr_start, p - addr_start);
        addr[p - addr_start] = '\0';

        char syscom[256];
        sprintf(syscom, "addr2line -e %s %s", filename, addr);
        // last parameter is the file name of the symbol
        system(syscom);
    }
}

#define cudaCheckErrors() \
    { cudaCheckErrorsFn(__FILE__, __LINE__); }

inline void cudaCheckErrorsFn(const char* file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s %s %d\n", cudaGetErrorString(error), file, line);
        print_trace();
        exit(-1);
    }
}

#endif  // _HASH_H_

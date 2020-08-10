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
#include <mutex>

// ----------------------------------------------
// Constants
// ----------------------------------------------

// Constants for hash table
static constexpr uint64_t NUM_ELEMENTS = 1 << 21;  // 1M elements
static constexpr uint KEY_SIZE = 15;
static constexpr uint WORD_SIZE = 16;
static constexpr uint BATCH_SIZE = 10000;
static constexpr uint CPU_BATCH_SIZE = 10;

static constexpr uint BLOCK_SIZE = 256;

// Used for kernel calls that touch ALL elements of the hash table
static constexpr uint NUM_BLOCKS_ALL = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Used for kernel calls that touch a batch of elements
static constexpr uint NUM_BLOCKS_BATCH = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Constants for hash function
static constexpr uint32_t PRIME = 0x01000193;  //   16777619
static const uint32_t SEED = 0x811C9DC5;       // 2166136261

namespace hash {

// ----------------------------------------------
// Shared structures
// ----------------------------------------------

// Stores Hash table entry fields needed internally
struct HashEntryInternal {
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
    void init();

    // Insert entry into hash table
    void insert_entry(CpuHashEntry* user_entry);

    // Finds the entry in the hash table
    void find_entry(CpuHashEntry* user_entry);

    // Clears all entries from hash table
    void clear();

    void debug_print_entries();

    HashEntryInternal entries[NUM_ELEMENTS];
};

// ----------------------------------------------
// GPU Hash Table
// ----------------------------------------------
//

class GpuHashTable : public CudaMemory {
  public:
    HashEntryInternal entries[NUM_ELEMENTS];
};

// ----------------------------------------------
// Hybrid Hash Table
// ----------------------------------------------

struct HybridHashEntryBatch : CudaManagedMemory {
    HybridHashEntryBatch() {
        memset(&keys, 0, BATCH_SIZE * KEY_SIZE);
        memset(&locations, 0, BATCH_SIZE);
        memset(&words, 0, BATCH_SIZE & WORD_SIZE);
    }

    char keys[BATCH_SIZE][KEY_SIZE];
    uint32_t locations[BATCH_SIZE];
    char words[BATCH_SIZE][WORD_SIZE];
};

struct HybridHashEntryInternal {
    bool occupied;
    char key[KEY_SIZE];
};

class HybridHashTable {
  public:
    HybridHashTable();

    void insert_batch(HybridHashEntryBatch* entry_batch, uint num_entries);

    void find_batch(HybridHashEntryBatch* entry_batch, uint num_entries);

    // Clears all entries from hash table
    void clear();

    void debug_print_entries();

    uint32_t find_location(char key[KEY_SIZE]);

    HybridHashEntryInternal key_storage[NUM_ELEMENTS];
    GpuHashTable* word_storage;

    static constexpr uint MAX_STREAMS = 10;
    // A buffer will be needed for every stream
    HybridHashEntryBatch* batch_bufs[MAX_STREAMS];
    cudaStream_t streams[MAX_STREAMS];
    uint stream_count;
    std::mutex find_lock;
};

}  // namespace hash

// ----------------------------------------------
// Debugging Stuff
// ----------------------------------------------

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

inline void abort_with_trace() {
    print_trace();
    exit(-1);
}

#endif  // _HASH_H_

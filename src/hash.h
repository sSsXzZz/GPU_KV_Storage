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
static constexpr uint BATCH_SIZE = 2;

// Constants used for kernel invocation
static constexpr uint BLOCK_SIZE = 256;
static constexpr uint NUM_BLOCKS = (NUM_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Constants for hash function
static constexpr uint32_t PRIME = 0x01000193;  //   16777619
static const uint32_t SEED = 0x811C9DC5;       // 2166136261

struct hash_entry_t {
    bool occupied;
    char key[KEY_SIZE];
    char word[WORD_SIZE];
};

struct hash_table_t {
    hash_entry_t entries[NUM_ELEMENTS];
};

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

class HashEntry : public CudaManagedMemory {
  public:
    HashEntry() {
        std::memset(key, 0, sizeof(key));
        std::memset(word, 0, sizeof(word));
    }

    HashEntry(char* k) : HashEntry() {
        std::strcpy(key, k);
    }

    HashEntry(char* k, char* w) : HashEntry(k) {
        std::strcpy(word, w);
    }

    HashEntry(HashEntry& h) = delete;
    HashEntry(HashEntry&& h) = delete;

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
inline bool operator==(const HashEntry& a, const HashEntry& b) {
    return (std::memcmp(a.key, b.key, KEY_SIZE) == 0) && (std::memcmp(a.word, b.word, WORD_SIZE) == 0);
}
inline std::ostream& operator<<(std::ostream& outs, const HashEntry& entry) {
    return outs << "(" << entry.key << ", " << entry.word << ")";
}

class HashEntryBatch : public CudaManagedMemory {
  public:
    HashEntryBatch() {
        memset(entries, 0, sizeof(entries));
    }
    HashEntryBatch(HashEntryBatch& h) = delete;
    HashEntryBatch(HashEntryBatch&& h) = delete;

    HashEntry entries[BATCH_SIZE];

    // Returns the entry at the given index
    inline HashEntry& at(std::size_t i) {
        return entries[i];
    }
};

// TODO should this just inherit from HashEntry?
// Stores extra fields needed internally
class HashEntryInternal {
  public:
    bool occupied;
    char key[KEY_SIZE];
    char word[WORD_SIZE];
};

class HashTable : public CudaMemory {
  public:
    // Inserts the entry into the hash table
    __device__ void insert_entry(HashEntry* user_entry);

    // Finds the entry in the hash table. The word in the user_entry is set to the word found.
    __device__ void find_entry(HashEntry* user_entry);

    // Returns the pointer to the entry at the given index
    __device__ HashEntryInternal* get_entry(uint32_t index);

    HashEntryInternal entries[NUM_ELEMENTS];

  private:
};

void checkEntryEqual(HashEntry& in, HashEntry& out) {
    if (in == out) {
        std::cout << "Entries " << in << " and " << out << " are equal" << std::endl;
    } else {
        std::cout << "Entries " << in << " and " << out << " are NOT equal!" << std::endl;
    }
}

// ----------------------------------------------
// Debugging Stuff Below
// ----------------------------------------------
#include <execinfo.h>
void print_trace(void) {
    char **strings;
    size_t i, size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    strings = backtrace_symbols(array, size);
    for (i = 0; i < size; i++)
        printf("%s\n", strings[i]);
    puts("");
    free(strings);
}

#define cudaCheckErrors() { cudaCheckErrorsFn(__FILE__, __LINE__); }

inline void cudaCheckErrorsFn(const char *file, int line) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s %s %d\n", cudaGetErrorString(error), file, line);
        print_trace();
        exit(-1);
    }
}

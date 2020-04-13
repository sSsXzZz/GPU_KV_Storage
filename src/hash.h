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
    HashEntry(char* k, char* w) {
        std::memset(key, 0, sizeof(key));
        std::memset(word, 0, sizeof(word));

        std::strcpy(key, k);
        std::strcpy(word, w);
    }

    HashEntry(char* k) {
        std::memset(key, 0, sizeof(key));
        std::memset(word, 0, sizeof(word));

        std::strcpy(key, k);
    }

    HashEntry(HashEntry& h) = delete;
    HashEntry(HashEntry&& h) = delete;

    char key[KEY_SIZE];
    char word[WORD_SIZE];
};
inline bool operator==(const HashEntry& a, const HashEntry& b) {
    return (std::memcmp(a.key, b.key, KEY_SIZE) == 0) && (std::memcmp(a.word, b.word, WORD_SIZE) == 0);
}

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

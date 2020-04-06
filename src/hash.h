#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
int cudaConfigureCall(dim3 grid_size, dim3 block_size, unsigned shared_size = 0,
                      cudaStream_t stream = 0);
#endif
#include <math.h>
#include <iostream>

static constexpr uint64_t N = 1<<20; // 1M elements
static constexpr uint KEY_SIZE = 32;
static constexpr uint WORD_SIZE = 64;
static constexpr uint BLOCK_SIZE = 256;
static constexpr uint NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

static constexpr uint32_t PRIME = 0x01000193; //   16777619
static const uint32_t SEED  = 0x811C9DC5; // 2166136261

struct hash_entry_t {
    bool occupied;
    char key[KEY_SIZE];
    char word[WORD_SIZE];
};

struct hash_table_t {
    hash_entry_t entries[N];
};

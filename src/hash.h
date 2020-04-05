#include <math.h>
#include <iostream>

static constexpr int N = 1<<20; // 1M elements
static constexpr int KEY_SIZE = 32;
static constexpr int WORD_SIZE = 64;

static constexpr uint32_t PRIME = 0x01000193; //   16777619
static const uint32_t SEED  = 0x811C9DC5; // 2166136261

struct hash_entry {
    bool occupied;
    char key[KEY_SIZE];
    char word[WORD_SIZE];
};

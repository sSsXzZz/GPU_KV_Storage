#include <thread>
#include <vector>

#include "hash.h"
#include "utils.h"

__device__ __host__ inline uint32_t fnv1a(char oneByte, uint32_t hash = SEED) {
    return (oneByte ^ hash) * PRIME;
}

__device__ __host__ uint32_t hash_function(char* key, uint32_t hash = SEED)  // fnv hash
{
    const char* ptr = (const char*)key;
    for (int i = 0; i < KEY_SIZE; i++) {
        hash = fnv1a(*ptr++, hash);
    }
    return hash % NUM_ELEMENTS;
}

namespace hash {

// ----------------------------------------------
// CPU Hash Table
// ----------------------------------------------

void CpuHashTable::init() {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        HashEntryInternal* entry = &entries[i];
        std::memset(entry, 0, sizeof(HashEntryInternal));
    }
}

void CpuHashTable::insert_entry(CpuHashEntry* user_entry) {
    char* key = user_entry->key;
    char* word = user_entry->word;

    uint32_t hash_val = hash_function(key);
    // printf("insert hash_val: %u\n", hash_val);
    // printf("Request to insert element with key %s, word %s\n", user_entry->key, user_entry->word);

    HashEntryInternal* entry = &entries[hash_val];
    // TODO handle case when hash table is full
    while (entry->occupied) {
        // If the keys match, just overwrite the data
        if (std::memcmp(key, entry->key, KEY_SIZE) == 0) {
            break;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &entries[hash_val];
        // printf("insert hash_val: %u\n", hash_val);
    }

    std::memcpy(entry->key, key, KEY_SIZE);
    std::memcpy(entry->word, word, WORD_SIZE);
    entry->occupied = true;
    // printf("Inserted element at hash_val %u, key %s, word %s\n", hash_val, entry->key, entry->word);
}

void CpuHashTable::find_entry(CpuHashEntry* user_entry) {
    char* key = user_entry->key;

    uint32_t hash_val = hash_function(key);
    // printf("get hash_val: %u\n", hash_val);

    HashEntryInternal* entry = &entries[hash_val];
    // Loop until we reach an empty entry OR find the key
    while (entry->occupied) {
        // //printf("Comparing keys %s & %s\n", entry->key, key);
        if (std::memcmp(key, entry->key, KEY_SIZE) == 0) {
            // printf("Found word: %s\n", entry->word);
            std::memcpy(user_entry->word, entry->word, WORD_SIZE);
            return;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &entries[hash_val];
        // printf("get hash_val: %u\n", hash_val);
    }

    // key not found, make sure the word we send back is empty
    user_entry->word[0] = 0;
    return;
}

void CpuHashTable::debug_print_entries() {
    printf("_____ ALL CPU ENTRIES _____\n");
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        HashEntryInternal* entry = &entries[i];
        if (entry->occupied) {
            printf("%u: (%s, %s)\n", i, entry->key, entry->word);
        }
    }
    printf("___________________________\n");
}

// ----------------------------------------------
// Hybrid/GPU Hash Table
// ----------------------------------------------

__global__ void gpu_init(GpuHashTable * hash_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < NUM_ELEMENTS; i += stride) {
        std::memset(&hash_table->words[i], 0, WORD_SIZE);
    }
}

__global__ void gpu_insert_batch(GpuHashTable* hash_table, HybridHashEntryBatch* entry_batch,
                                           uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        char* word = entry_batch->words[i];
        uint32_t location = entry_batch->locations[i];

        std::memcpy(&hash_table->words[location], word, WORD_SIZE);
    }
}

__global__ void gpu_find_batch(GpuHashTable* hash_table, HybridHashEntryBatch* entry_batch,
                                         uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        uint32_t location = entry_batch->locations[i];

        std::memcpy(&entry_batch->words[i], &hash_table->words[location], WORD_SIZE);
    }
}

HybridHashTable::HybridHashTable() {
    word_storage = new GpuHashTable;
    cudaDeviceSynchronize();
    cudaCheckErrors();

    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        std::memset(&key_storage[i], 0, sizeof(HybridHashEntryInternal));
    }

    gpu_init<<<NUM_BLOCKS_ALL, BLOCK_SIZE>>>(word_storage);
    cudaDeviceSynchronize();
    cudaCheckErrors();
}

uint32_t HybridHashTable::find_location(char key[KEY_SIZE]) {
    uint32_t hash_val = hash_function(key);

    HybridHashEntryInternal* entry = &key_storage[hash_val];
    // TODO handle case when hash table is full
    while (entry->occupied) {
        // If the keys match, just overwrite the data
        if (std::memcmp(key, entry->key, KEY_SIZE) == 0) {
            break;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &key_storage[hash_val];
    }

    return hash_val;
}

void HybridHashTable::insert_batch(HybridHashEntryBatch* entry_batch, uint num_entries) {
    //time_t t0 = get_time_us();
    // iterate through entries, find the key location, and populate the internal_batch with the locations
    for (uint i = 0; i < num_entries; i++) {
        uint32_t location = find_location(entry_batch->keys[i]);
        entry_batch->locations[i] = location;

        // Set key storage to occupied and copy key
        key_storage[location].occupied = true;
        std::memcpy(key_storage[location].key, entry_batch->keys[i], KEY_SIZE);
    }
    //printf("%lu us to find hashes\n", get_time_us() - t0);

    // TODO change this to transfer only words
    gpu_insert_batch<<<NUM_BLOCKS_BATCH, BLOCK_SIZE>>>(word_storage, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

void HybridHashTable::find_batch(HybridHashEntryBatch* entry_batch, uint num_entries) {
    // iterate through entries, find the key location, and populate the internal_batch with the location
    for (uint i = 0; i < num_entries; i++) {
        entry_batch->locations[i] = find_location(entry_batch->keys[i]);
    }

    gpu_find_batch<<<NUM_BLOCKS_BATCH, BLOCK_SIZE>>>(word_storage, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

void HybridHashTable::debug_print_entries() {
    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        HybridHashEntryInternal* entry = &key_storage[i];
        if (entry->occupied) {
            printf("%u: (key %s)\n", i, entry->key);
        }
    }
}

}  // namespace hash

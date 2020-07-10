#include <thread>
#include <vector>

#include "hash.h"

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

// ----------------------------------------------
// GPU Hash Table
// ----------------------------------------------

__device__ bool device_memcmp(char* word1, char* word2, uint len) {
    for (uint i = 0; i < len; i++) {
        if (word1[i] != word2[i]) {
            return false;
        }
    }
    return true;
}

namespace hash {
__device__ void GpuHashTable::insert_entry(GpuHashEntry* user_entry) {
    char* key = user_entry->key;
    char* word = user_entry->word;

    uint32_t hash_val = hash_function(key);
    // printf("insert hash_val: %u\n", hash_val);
    // printf("Request to insert element with key %s, word %s\n", user_entry->key, user_entry->word);

    HashEntryInternal* entry = &entries[hash_val];
    // TODO handle case when hash table is full
    while (entry->occupied) {
        // If the keys match, just overwrite the data
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
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

__device__ void GpuHashTable::find_entry(GpuHashEntry* user_entry) {
    char* key = user_entry->key;

    uint32_t hash_val = hash_function(key);
    // printf("get hash_val: %u\n", hash_val);

    HashEntryInternal* entry = &entries[hash_val];
    // Loop until we reach an empty entry OR find the key
    while (entry->occupied) {
        // //printf("Comparing keys %s & %s\n", entry->key, key);
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
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

__device__ HashEntryInternal* GpuHashTable::get_entry(uint32_t index) {
    return &entries[index];
}

__device__ __host__ inline void init_hash_entry(HashEntryInternal* entry) {
    std::memset(entry, 0, sizeof(HashEntryInternal));
}

__global__ void init_hash_table_internal(GpuHashTable* hash_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < NUM_ELEMENTS; i += stride) {
        HashEntryInternal* entry = hash_table->get_entry(i);
        init_hash_entry(entry);
    }
}

__global__ void hash_insert_internal(GpuHashTable* hash_table, GpuHashEntry* entry) {
    hash_table->insert_entry(entry);
}

__global__ void hash_find_internal(GpuHashTable* hash_table, GpuHashEntry* entry) {
    hash_table->find_entry(entry);
}

__global__ void hash_insert_batch_internal(GpuHashTable* hash_table, GpuHashEntryBatch* entry_batch, uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        // printf("Batch insert index %u\n", i);
        hash_table->insert_entry(&entry_batch->entries[i]);
    }
}

__global__ void hash_find_batch_internal(GpuHashTable* hash_table, GpuHashEntryBatch* entry_batch, uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        hash_table->find_entry(&entry_batch->entries[i]);
    }
}

void init_hash_table(GpuHashTable* hash_table) {
    init_hash_table_internal<<<NUM_BLOCKS_ALL, BLOCK_SIZE>>>(hash_table);
    cudaDeviceSynchronize();
}

void hash_insert(GpuHashTable* hash_table, GpuHashEntry* entry) {
    hash_insert_internal<<<1, 1>>>(hash_table, entry);
    cudaDeviceSynchronize();
}

void hash_find(GpuHashTable* hash_table, GpuHashEntry* entry) {
    hash_find_internal<<<1, 1>>>(hash_table, entry);
    cudaDeviceSynchronize();
}

void hash_insert_batch(GpuHashTable* hash_table, GpuHashEntryBatch* entry_batch, uint num_entries) {
    hash_insert_batch_internal<<<NUM_BLOCKS_BATCH, BLOCK_SIZE>>>(hash_table, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

void hash_find_batch(GpuHashTable* hash_table, GpuHashEntryBatch* entry_batch, uint num_entries) {
    hash_find_batch_internal<<<NUM_BLOCKS_BATCH, BLOCK_SIZE>>>(hash_table, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

// ----------------------------------------------
// CPU Hash Table
// ----------------------------------------------

void CpuHashTable::init() {
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        HashEntryInternal* entry = &entries[i];
        init_hash_entry(entry);
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
// Hybrid Hash Table
// ----------------------------------------------

HybridHashTable::HybridHashTable() {
    word_storage = new GpuHashTable;
    cudaDeviceSynchronize();
    cudaCheckErrors();

    for (uint32_t i = 0; i < NUM_ELEMENTS; i++) {
        std::memset(&key_storage[i], 0, sizeof(HybridHashEntryInternal));
    }
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
    // iterate through entries, find the key location, and populate the internal_batch with the locations
    for (uint i = 0; i < num_entries; i++) {
        uint32_t location = find_location(entry_batch->keys[i]);
        entry_batch->locations[i] = location;

        // Set key storage to occupied and copy key
        key_storage[location].occupied = true;
        std::memcpy(key_storage[location].key, entry_batch->keys[i], KEY_SIZE);
    }

    // TODO change this to work only words
    hash_insert_batch(word_storage, entry_batch, num_entries);
}

void HybridHashTable::find_batch(HybridHashEntryBatch* entry_batch, uint num_entries) {
    // iterate through entries, find the key location, and populate the internal_batch with the location
    for (uint i = 0; i < num_entries; i++) {
        entry_batch->locations[i] = find_location(entry_batch->keys[i]);
    }

    hash_find_batch(word_storage, entry_batch, num_entries);
}

__global__ void hash_insert_batch_internal(GpuHashTable* hash_table, HybridHashEntryBatch* entry_batch,
                                           uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        char* word = entry_batch->words[i];
        uint32_t location = entry_batch->locations[i];

        std::memcpy(&hash_table->entries[location].word, word, WORD_SIZE);
    }
}

__global__ void hash_find_batch_internal(GpuHashTable* hash_table, HybridHashEntryBatch* entry_batch,
                                         uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        uint32_t location = entry_batch->locations[i];

        std::memcpy(&entry_batch->words[i], &hash_table->entries[location].word, WORD_SIZE);
    }
}

void hash_insert_batch(GpuHashTable* hash_table, HybridHashEntryBatch* entry_batch, uint num_entries) {
    hash_insert_batch_internal<<<NUM_BLOCKS_BATCH, BLOCK_SIZE>>>(hash_table, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

void hash_find_batch(GpuHashTable* hash_table, HybridHashEntryBatch* entry_batch, uint num_entries) {
    hash_find_batch_internal<<<NUM_BLOCKS_BATCH, BLOCK_SIZE>>>(hash_table, entry_batch, num_entries);
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

// ----------------------------------------------
// Debugging
// ----------------------------------------------
__global__ void print_all_entries_internal(GpuHashTable* hash_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < NUM_ELEMENTS; i += stride) {
        HashEntryInternal* entry = hash_table->get_entry(i);
        if (entry->occupied) {
            printf("%u: (%s, %s)\n", i, entry->key, entry->word);
        }
    }
}

void print_all_entries(GpuHashTable* hash_table) {
    printf("_____ ALL ENTRIES _____\n");
    print_all_entries_internal<<<NUM_BLOCKS_ALL, BLOCK_SIZE>>>(hash_table);
    cudaDeviceSynchronize();
    printf("_______________________\n");
}

}  // namespace hash

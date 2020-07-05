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

__device__ inline void init_hash_entry(HashEntryInternal* entry) {
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

// TODO temporary insert & find functions
void CpuHashTable::insert_batch(CpuHashEntryBatch* entry_batch, uint num_entries) {
    for (int i = 0; i < num_entries; i++) {
        insert_entry(&entry_batch->entries[i]);
    }
}

void CpuHashTable::find_batch(CpuHashEntryBatch* entry_batch, uint num_entries) {
    for (int i = 0; i < num_entries; i++) {
        find_entry(&entry_batch->entries[i]);
    }
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

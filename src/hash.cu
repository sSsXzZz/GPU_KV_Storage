#include "hash.h"

__device__ inline uint32_t fnv1a(char oneByte, uint32_t hash = SEED) {
    return (oneByte ^ hash) * PRIME;
}

__device__ uint32_t hash_function(char* key, uint32_t hash = SEED)  // fnv hash
{
    const char* ptr = (const char*)key;
    for (int i = 0; i < KEY_SIZE; i++) {
        hash = fnv1a(*ptr++, hash);
    }
    return hash % NUM_ELEMENTS;
}

__device__ bool device_memcmp(char* word1, char* word2, uint len) {
    for (uint i = 0; i < len; i++) {
        if (word1[i] != word2[i]) {
            return false;
        }
    }
    return true;
}
__device__ void HashTable::insert_entry(HashEntry* user_entry) {
    char* key = user_entry->key;
    char* word = user_entry->word;

    uint32_t hash_val = hash_function(key);
    printf("insert hash_val: %u\n", hash_val);
    printf("Request to insert element with key %s, word %s\n", user_entry->key, user_entry->word);

    HashEntryInternal* entry = &entries[hash_val];
    // TODO handle case when hash table is full
    while (entry->occupied) {
        // If the keys match, just overwrite the data
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
            break;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &entries[hash_val];
        printf("insert hash_val: %u\n", hash_val);
    }

    std::memcpy(entry->key, key, KEY_SIZE);
    std::memcpy(entry->word, word, WORD_SIZE);
    entry->occupied = true;
    printf("Inserted element at hash_val %u, key %s, word %s\n", hash_val, entry->key, entry->word);
}

__device__ void HashTable::find_entry(HashEntry* user_entry) {
    char* key = user_entry->key;

    uint32_t hash_val = hash_function(key);
    printf("get hash_val: %u\n", hash_val);

    HashEntryInternal* entry = &entries[hash_val];
    // Loop until we reach an empty entry OR find the key
    while (entry->occupied) {
        // printf("Comparing keys %s & %s\n", entry->key, key);
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
            printf("Found word: %s\n", entry->word);
            std::memcpy(user_entry->word, entry->word, WORD_SIZE);
            return;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &entries[hash_val];
        printf("get hash_val: %u\n", hash_val);
    }

    // key not found, make sure the word we send back is empty
    user_entry->word[0] = 0;
    return;
}

__device__ HashEntryInternal* HashTable::get_entry(uint32_t index) {
    return &entries[index];
}

__device__ inline void init_hash_entry(HashEntryInternal* entry) {
    std::memset(entry, 0, sizeof(HashEntryInternal));
}

__global__ void init_hash_table_internal(HashTable* hash_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < NUM_ELEMENTS; i += stride) {
        HashEntryInternal* entry = hash_table->get_entry(i);
        init_hash_entry(entry);
    }
}

__global__ void hash_insert_internal(HashTable* hash_table, HashEntry* entry) {
    hash_table->insert_entry(entry);
}

__global__ void hash_find_internal(HashTable* hash_table, HashEntry* entry) {
    hash_table->find_entry(entry);
}

__global__ void hash_insert_batch_internal(HashTable* hash_table, HashEntryBatch* entry_batch, uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        printf("Batch insert index %u\n", i);
        hash_table->insert_entry(&entry_batch->entries[i]);
    }
}

__global__ void hash_find_batch_internal(HashTable* hash_table, HashEntryBatch* entry_batch, uint num_entries) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < num_entries; i += stride) {
        hash_table->find_entry(&entry_batch->entries[i]);
    }
}

void init_hash_table(HashTable* hash_table) {
    init_hash_table_internal<<<NUM_BLOCKS, BLOCK_SIZE>>>(hash_table);
    cudaDeviceSynchronize();
}

void hash_insert(HashTable* hash_table, HashEntry* entry) {
    hash_insert_internal<<<1, 1>>>(hash_table, entry);
    cudaDeviceSynchronize();
}

void hash_find(HashTable* hash_table, HashEntry* entry) {
    hash_find_internal<<<1, 1>>>(hash_table, entry);
    cudaDeviceSynchronize();
}

void hash_insert_batch(HashTable* hash_table, HashEntryBatch* entry_batch, uint num_entries) {
    hash_insert_batch_internal<<<NUM_BLOCKS, BLOCK_SIZE>>>(hash_table, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

void hash_find_batch(HashTable* hash_table, HashEntryBatch* entry_batch, uint num_entries) {
    hash_find_batch_internal<<<NUM_BLOCKS, BLOCK_SIZE>>>(hash_table, entry_batch, num_entries);
    cudaDeviceSynchronize();
}

// Used for debugging only
__global__ void print_all_entries_internal(HashTable* hash_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < NUM_ELEMENTS; i += stride) {
        HashEntryInternal* entry = hash_table->get_entry(i);
        if (entry->occupied) {
            printf("%u: (%s, %s)\n", i, entry->key, entry->word);
        }
    }
}

void print_all_entries(HashTable* hash_table) {
    printf("_____ ALL ENTRIES _____\n");
    print_all_entries_internal<<<NUM_BLOCKS, BLOCK_SIZE>>>(hash_table);
    cudaDeviceSynchronize();
    printf("_______________________\n");
}

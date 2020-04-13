#include "hash.h"

__device__ void init_hash_entry(hash_entry_t* entry) {
    std::memset(entry, 0, sizeof(hash_entry_t));
    entry->occupied = false;  // redundant but just to be certain
}

__global__ void init_hash_table(int n, hash_table_t* hash_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        hash_entry_t entry = hash_table->entries[i];
        init_hash_entry(&entry);
    }
}

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

    HashEntryInternal* entry = &entries[hash_val];
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
}

__device__ void HashTable::find_entry(HashEntry* user_entry) {
    char* key = user_entry->key;

    uint32_t hash_val = hash_function(key);
    printf("get hash_val: %u\n", hash_val);

    HashEntryInternal* entry = &entries[hash_val];
    // Loop until we reach an empty entry OR find the key
    while (entry->occupied) {
        printf("Comparing keys %s & %s\n", entry->key, key);
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

__device__ void init_hash_entry(HashEntryInternal* entry) {
    std::memset(entry, 0, sizeof(HashEntryInternal));
}

__global__ void init_hash_table(HashTable* hash_table, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uint32_t i = index; i < n; i += stride) {
        HashEntryInternal* entry = hash_table->get_entry(i);
        init_hash_entry(entry);
    }
}

__global__ void hash_insert(HashTable* hash_table, HashEntry* entry) {
    hash_table->insert_entry(entry);
}

__global__ void hash_find(HashTable* hash_table, HashEntry* entry) {
    hash_table->find_entry(entry);
}

__global__ void insert_entry(hash_table_t* hash_table, char* key, char* word) {
    uint32_t hash_val = hash_function(key);
    printf("insert hash_val: %u\n", hash_val);
    hash_entry_t* entry;

    entry = &hash_table->entries[hash_val];
    while (entry->occupied) {
        // If the keys match, just overwrite the data
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
            break;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &hash_table->entries[hash_val];
        printf("insert hash_val: %u\n", hash_val);
    }
    memcpy(entry->key, key, KEY_SIZE);
    memcpy(entry->word, word, WORD_SIZE);
    entry->occupied = true;
}

__global__ void get_entry(hash_table_t* hash_table, char* key, char* host_word) {
    uint32_t hash_val = hash_function(key);
    printf("get hash_val: %u\n", hash_val);
    hash_entry_t* entry;

    entry = &hash_table->entries[hash_val];
    // Loop until we reach an empty entry OR find the key
    while (entry->occupied) {
        printf("Comparing keys %s & %s\n", entry->key, key);
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
            printf("Found word: %s\n", entry->word);
            memcpy(host_word, entry->word, WORD_SIZE);
            return;
        }
        hash_val = (hash_val + 1) % NUM_ELEMENTS;
        entry = &hash_table->entries[hash_val];
        printf("get hash_val: %u\n", hash_val);
    }
    host_word = nullptr;
    return;
}

void test_old_table() {
    hash_table_t* hash_table;

    cudaMallocManaged(&hash_table, sizeof(hash_table_t));

    init_hash_table<<<NUM_BLOCKS, BLOCK_SIZE>>>(NUM_ELEMENTS, hash_table);
    cudaDeviceSynchronize();

    char *key, *word;
    cudaMallocManaged(&key, KEY_SIZE);
    cudaMallocManaged(&word, WORD_SIZE);
    strcpy(key, "abcdefghijklmnopqrstuvwxyz12345");
    strcpy(word, "abcdefghijklmnopqrstuvwxyz12345abcdefghijklmnopqrstuvwxyz123456");

    insert_entry<<<1, 1>>>(hash_table, key, word);
    cudaDeviceSynchronize();

    char* buffer;
    cudaMallocManaged(&buffer, WORD_SIZE);
    get_entry<<<1, 1>>>(hash_table, key, buffer);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    if (memcmp(buffer, word, WORD_SIZE) == 0) {
        std::cout << "Hooray they're equal" << std::endl;
        std::cout << "Word: " << word << std::endl;
        std::cout << "Buffer: " << buffer << std::endl;
    } else {
        std::cout << "Boo I suck" << std::endl;
        std::cout << "Word: " << word << std::endl;
        std::cout << "Buffer: " << buffer << std::endl;
    }

    cudaFree(hash_table);
    cudaFree(key);
    cudaFree(word);
    cudaFree(buffer);
}

void test_new_table() {
    HashTable* h =  new HashTable;
    cudaDeviceSynchronize();

    init_hash_table<<<NUM_BLOCKS, BLOCK_SIZE>>>(h, NUM_ELEMENTS);
    cudaDeviceSynchronize();

    char test_key[] = "abcdefghijklmnopqrstuvwxyz12345";
    char test_word[] = "abcdefghijklmnopqrstuvwxyz12345abcdefghijklmnopqrstuvwxyz123456";

    HashEntry* in = new HashEntry(test_key, test_word);
    HashEntry* out = new HashEntry(test_key);
    cudaDeviceSynchronize();

    hash_insert<<<1, 1>>>(h, in);
    cudaDeviceSynchronize();
    hash_find<<<1, 1>>>(h, out);
    cudaDeviceSynchronize();
    if (*in == *out) {
        std::cout << "Hooray they're equal" << std::endl;
    } else {
        std::cout << "Boo I suck" << std::endl;
    }

    delete h;
    delete in;
    delete out;
}

int main(void) {
    test_new_table();

    return 0;
}

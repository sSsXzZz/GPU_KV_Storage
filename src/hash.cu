#include "hash.h"
#include <cstdio>
#include <cstddef>

// function to add the elements of two arrays
/*
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}
*/

__device__
void init_hash_entry(hash_entry *entry)
{
    memset(entry, 0, sizeof(hash_entry));
    entry->occupied = false; // redundant but just to be certain
}

__global__
void init_hash_table(int n, char *hash_table)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        hash_entry *entry = (hash_entry*) &hash_table[i];
        init_hash_entry(entry);
    }
}

__device__
inline uint32_t fnv1a(char oneByte, uint32_t hash = SEED)
{
    return (oneByte ^ hash) * PRIME;
}

__device__
uint32_t hash_function(char *key, uint32_t hash = SEED) //fnv hash
{
    const char* ptr = (const char*)key;
    for (int i = 0; i < KEY_SIZE; i++) {
        hash = fnv1a(*ptr++, hash);
    }
    return hash % N;
}

__device__
bool device_memcmp(char* word1, char* word2, uint len)
{
    for (uint i=0; i < len; i++) {
        if (word1[i] != word2[i]) {
            return false;
        }
    }
    return true;
}

__global__
void insert_entry(char *hash_table, char *key, char *word)
{
    uint32_t hash_val = hash_function(key);
    printf("insert hash_val: %u\n", hash_val);
    hash_entry *entry;

    entry = (hash_entry*) &hash_table[hash_val];
    while (entry->occupied) {
        // If the keys match, just overwrite the data
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
            break;
        }
        hash_val =  (hash_val + 1) % N;
        entry = (hash_entry*) &hash_table[hash_val];
        printf("insert hash_val: %u\n", hash_val);
    }
    memcpy(entry->key, key, KEY_SIZE);
    memcpy(entry->word, word, WORD_SIZE);
    entry->occupied = true;
}

__global__
void get_entry(char *hash_table, char *key, char *host_word)
{
    uint32_t hash_val = hash_function(key);
    printf("get hash_val: %u\n", hash_val);
    hash_entry *entry;

    entry = (hash_entry*) &hash_table[hash_val];
    // Loop until we reach an empty entry OR find the key
    while (entry->occupied) {
        printf("Comparing keys %s & %s\n", entry->key, key);
        if (device_memcmp(key, entry->key, KEY_SIZE)) {
            printf("Found word: %s\n", entry->word);
            memcpy(host_word, entry->word, WORD_SIZE);
            return;
        }
        hash_val =  (hash_val + 1) % N;
        entry = (hash_entry*) &hash_table[hash_val];
        printf("get hash_val: %u\n", hash_val);
    }
    host_word = nullptr;
    return;
}

int main(void)
{
    char *hash_table;

    cudaMallocManaged(&hash_table, N*sizeof(hash_entry));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    init_hash_table<<<numBlocks, blockSize>>>(N, hash_table);
    cudaDeviceSynchronize();

    // TODO run a routine here to test the hash table
    char *key, *word;
    cudaMallocManaged(&key, KEY_SIZE);
    cudaMallocManaged(&word, WORD_SIZE);
    strcpy(key, "abcdefghijklmnopqrstuvwxyz12345");
    strcpy(word, "abcdefghijklmnopqrstuvwxyz12345abcdefghijklmnopqrstuvwxyz123456");

    insert_entry<<<1, 1>>>(hash_table, key, word);
    cudaDeviceSynchronize();

    char *buffer;
    cudaMallocManaged(&buffer, WORD_SIZE);
    get_entry<<<1, 1>>>(hash_table, key, buffer);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
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

    /*std::byte b;*/
    /*(void)(b);*/

    // Free memory
    cudaFree(hash_table);

    return 0;
}

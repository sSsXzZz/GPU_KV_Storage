#include <chrono>
#include <ctime>
#include <random>
#include <unordered_map>

#include "hash.h"

using UniformDistribution = std::uniform_int_distribution<uint>;
using Generator = std::mt19937;
using DataMap = std::unordered_map<std::string, std::string>;
using CLOCK = std::chrono::high_resolution_clock;

static constexpr char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static constexpr uint NUM_TEST_ENTRIES = 10000;

void checkEntryEqual(GpuHashEntry& in, GpuHashEntry& out) {
    if (in == out) {
        // std::cout << "Entries " << in << " and " << out << " are equal" << std::endl;
    } else {
        std::cout << "Entries " << in << " and " << out << " are NOT equal!" << std::endl;
        abort_with_trace();
    }
}

std::string get_random_string(UniformDistribution& char_picker, Generator& generator, uint used_space, uint length) {
    // This will make sure our string is already null terminated
    std::string s(length, 0);
    for (uint i = 0; i < used_space; i++) {
        s[i] = alphanum[char_picker(generator)];
    }
    return s;
}

// TODO use test data here?
void checkGpuBatchedOutput(DataMap& test_data, GpuHashEntryBatch* out_batch, uint num_entries) {
    for (uint i = 0; i < num_entries; i++) {
        const char* out_key = out_batch->entries[i].key;
        const char* out_word = out_batch->entries[i].word;

        std::string key(out_key, KEY_SIZE);
        auto it = test_data.find(key);
        if (it == test_data.end()) {
            printf("Key %s not found in the test data map\n", key.c_str());
            abort_with_trace();
        }
        std::string word = test_data[key];

        if (strcmp(key.c_str(), out_key) == 0 && strcmp(word.c_str(), out_word) == 0) {
            // printf("entry(%s, %s) == map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
        } else {
            printf("GPU: entry(%s, %s) != map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
            abort_with_trace();
        }
    }
}

void checkCpuOutput(DataMap& test_data, CpuHashEntry* out_entry) {
    const char* out_key = out_entry->key;
    const char* out_word = out_entry->word;

    std::string key(out_key, KEY_SIZE);
    std::string word = test_data[key];
    if (strcmp(key.c_str(), out_key) == 0 && strcmp(word.c_str(), out_word) == 0) {
        // printf("entry(%s, %s) == map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
    } else {
        printf("CPU: entry(%s, %s) != map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
        abort_with_trace();
    }
}

time_t ts_diff_us(timespec t0, timespec t1) {
    return (((t1.tv_sec - t0.tv_sec) * 1000000000) + (t1.tv_nsec - t0.tv_nsec)) / 1000;
}

void test_cpu_table(DataMap& test_entries) {
    CpuHashTable* h = new CpuHashTable;
    h->init();

    CpuHashEntryBatch* in_batch = new CpuHashEntryBatch;
    CpuHashEntryBatch* out_batch = new CpuHashEntryBatch;

    CpuHashEntry* in_entry = new CpuHashEntry;
    CpuHashEntry* out_entry = new CpuHashEntry;

    // Iterate through map and batch insert entries
    struct timespec t0_insert;
    clock_gettime(CLOCK_MONOTONIC, &t0_insert);
    for (auto& entry : test_entries) {
        const std::string& key = entry.first;
        const std::string& word = entry.second;

        std::memcpy(&in_entry->key, key.c_str(), key.size());
        std::memcpy(&in_entry->word, word.c_str(), word.size());

        if (in_entry->key[0] == '\0') {
            std::cout << "FOUND THE EMPTY KEY\n";
        }
        h->insert_entry(in_entry);
    }
    struct timespec t1_insert;
    clock_gettime(CLOCK_MONOTONIC, &t1_insert);
    time_t tdiff_insert = ts_diff_us(t0_insert, t1_insert);
    std::cout << "CPU Insert time: " << tdiff_insert << " us" << std::endl;

    // h->debug_print_entries();

    // Iterate through map & batch find entries
    struct timespec t0_find;
    clock_gettime(CLOCK_MONOTONIC, &t0_find);
    for (auto& entry : test_entries) {
        const std::string& key = entry.first;

        std::memcpy(&out_entry->key, key.c_str(), key.size());
        std::memset(&out_entry->word, 0, WORD_SIZE);

        h->find_entry(out_entry);

        if (out_entry->key[0] == '\0') {
            std::cout << "FOUND THE EMPTY KEY\n";
        }
        checkCpuOutput(test_entries, out_entry);
    }
    struct timespec t1_find;
    clock_gettime(CLOCK_MONOTONIC, &t1_find);
    time_t tdiff_find = ts_diff_us(t0_find, t1_find);
    std::cout << "CPU Find time: " << tdiff_find << " us" << std::endl;

    // print_all_entries(h);

    // std::cout << "_____ UNORDED_MAP ENTRIES _____\n";
    for (auto entry : test_entries) {
        // printf("[%s]=%s\n", entry.first.c_str(), entry.second.c_str());
    }
    // std::cout << "__________________________\n";

    delete h;
    delete in_batch;
    delete out_batch;
}

void test_gpu_table(DataMap& test_entries) {
    GpuHashTable* h = new GpuHashTable;
    init_hash_table(h);
    cudaCheckErrors();

    GpuHashEntryBatch* in_batch = new GpuHashEntryBatch;
    GpuHashEntryBatch* out_batch = new GpuHashEntryBatch;
    cudaDeviceSynchronize();
    cudaCheckErrors();

    // Iterate through map and batch insert entries
    struct timespec t0_insert;
    clock_gettime(CLOCK_MONOTONIC, &t0_insert);
    int batch_index = 0;  // Count of entries in this batch
    for (auto& entry : test_entries) {
        const std::string& key = entry.first;
        const std::string& word = entry.second;
        std::memcpy(&in_batch->entries[batch_index].key, key.c_str(), key.size());
        std::memcpy(&in_batch->entries[batch_index].word, word.c_str(), word.size());
        batch_index++;

        // Insert entries using max BATCH_SIZE
        if (batch_index == BATCH_SIZE) {
            hash_insert_batch(h, in_batch, BATCH_SIZE);
            cudaCheckErrors();
            batch_index = 0;
        }
    }
    // Insert remaining entries
    if (batch_index > 0 && batch_index < BATCH_SIZE) {
        hash_insert_batch(h, in_batch, batch_index);
        cudaCheckErrors();
    }
    struct timespec t1_insert;
    clock_gettime(CLOCK_MONOTONIC, &t1_insert);
    time_t tdiff_insert = ts_diff_us(t0_insert, t1_insert);
    std::cout << "GPU Insert time: " << tdiff_insert << " us" << std::endl;

    // Iterate through map & batch find entries
    struct timespec t0_find;
    clock_gettime(CLOCK_MONOTONIC, &t0_find);
    batch_index = 0;
    for (auto& entry : test_entries) {
        const std::string& key = entry.first;
        std::memcpy(&out_batch->entries[batch_index].key, key.c_str(), key.size());
        std::memset(&out_batch->entries[batch_index].word, 0, WORD_SIZE);
        batch_index++;

        // Insert entries using max BATCH_SIZE
        if (batch_index == BATCH_SIZE) {
            hash_find_batch(h, out_batch, BATCH_SIZE);
            cudaCheckErrors();

            checkGpuBatchedOutput(test_entries, out_batch, BATCH_SIZE);
            batch_index = 0;
        }
    }
    // Check remaining entries
    if (batch_index > 0 && batch_index < BATCH_SIZE) {
        hash_find_batch(h, out_batch, BATCH_SIZE);
        cudaCheckErrors();

        checkGpuBatchedOutput(test_entries, out_batch, batch_index);
    }
    struct timespec t1_find;
    clock_gettime(CLOCK_MONOTONIC, &t1_find);
    time_t tdiff_find = ts_diff_us(t0_find, t1_find);
    std::cout << "GPU Find time: " << tdiff_find << " us" << std::endl;

    // print_all_entries(h);

    // std::cout << "_____ UNORDED_MAP ENTRIES _____\n";
    for (auto& entry : test_entries) {
        // printf("[%s]=%s\n", entry.first.c_str(), entry.second.c_str());
    }
    // std::cout << "__________________________\n";

    delete h;
    delete in_batch;
    delete out_batch;
}

int main(void) {
    Generator generator(std::chrono::system_clock::now().time_since_epoch().count());

    UniformDistribution char_picker(0, sizeof(alphanum) - 2);  // -1 for null terminator, -1 for 0 index
    UniformDistribution key_size(1, KEY_SIZE - 1);
    UniformDistribution word_size(1, WORD_SIZE - 1);

    // Generate random key, value pairs
    DataMap test_entries;
    for (uint i = 0; i < NUM_TEST_ENTRIES; i++) {
        std::string key = get_random_string(char_picker, generator, key_size(generator), KEY_SIZE);
        std::string word = get_random_string(char_picker, generator, word_size(generator), WORD_SIZE);

        // std::cout << "Generated entry (" << key << ", " << word << ")" << std::endl;

        test_entries[key] = word;
    }

    test_gpu_table(test_entries);
    test_cpu_table(test_entries);

    return 0;
}

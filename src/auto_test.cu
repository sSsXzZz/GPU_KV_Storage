#include <chrono>
#include <ctime>
#include <random>
#include <unordered_map>

#include "hash.h"
#include "utils.h"

using hash::CpuHashEntry;
using hash::CpuHashEntryBatch;
using hash::CpuHashTable;
using hash::GpuHashEntry;
using hash::GpuHashEntryBatch;
using hash::GpuHashTable;
using hash::HybridHashEntryBatch;
using hash::HybridHashTable;

using UniformDistribution = std::uniform_int_distribution<uint>;
using Generator = std::mt19937;
using DataMap = std::unordered_map<std::string, std::string>;

static constexpr char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static constexpr uint NUM_TEST_ENTRIES = 1'000'000;

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

void checkHybridBatchedOutput(DataMap& test_data, HybridHashEntryBatch* out_batch, uint num_entries) {
    for (uint i = 0; i < num_entries; i++) {
        const char* out_key = out_batch->keys[i];
        const char* out_word = out_batch->words[i];

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

void test_cpu_table(DataMap& test_entries) {
    CpuHashTable* h = new CpuHashTable;
    h->init();

    CpuHashEntryBatch* in_batch = new CpuHashEntryBatch;
    CpuHashEntryBatch* out_batch = new CpuHashEntryBatch;

    CpuHashEntry* in_entry = new CpuHashEntry;
    CpuHashEntry* out_entry = new CpuHashEntry;

    // Iterate through map and batch insert entries
    time_t t0_insert = get_time_us();
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
    time_t t1_insert = get_time_us();
    std::cout << "CPU Insert time: " << t1_insert - t0_insert << " us" << std::endl;

    // h->debug_print_entries();

    // Iterate through map & batch find entries
    time_t t0_find = get_time_us();
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
    time_t t1_find = get_time_us();
    std::cout << "CPU Find time: " << t1_find - t0_find << " us" << std::endl;

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

void test_hybrid_table(DataMap& test_entries) {
    HybridHashTable* h = new HybridHashTable;
    cudaCheckErrors();

    HybridHashEntryBatch* in_batch = new HybridHashEntryBatch;
    HybridHashEntryBatch* out_batch = new HybridHashEntryBatch;
    cudaDeviceSynchronize();
    cudaCheckErrors();

    // Iterate through map and batch insert entries
    time_t t0_insert = get_time_us();
    int batch_index = 0;  // Count of entries in this batch
    for (auto& entry : test_entries) {
        const std::string& key = entry.first;
        const std::string& word = entry.second;
        std::memcpy(&in_batch->keys[batch_index], key.c_str(), key.size());
        std::memcpy(&in_batch->words[batch_index], word.c_str(), word.size());
        batch_index++;

        // Insert entries using max BATCH_SIZE
        if (batch_index == BATCH_SIZE) {
            h->insert_batch(in_batch, BATCH_SIZE);
            cudaCheckErrors();
            batch_index = 0;
        }
    }
    // Insert remaining entries
    if (batch_index > 0 && batch_index < BATCH_SIZE) {
        h->insert_batch(in_batch, batch_index);
        cudaCheckErrors();
    }
    time_t t1_insert = get_time_us();
    std::cout << "Hybrid Insert time: " << t1_insert - t0_insert << " us" << std::endl;

    // h->debug_print_entries();

    // Iterate through map & batch find entries
    time_t t0_find = get_time_us();
    batch_index = 0;
    for (auto& entry : test_entries) {
        const std::string& key = entry.first;
        std::memcpy(&out_batch->keys[batch_index], key.c_str(), key.size());
        std::memset(&out_batch->words[batch_index], 0, WORD_SIZE);
        batch_index++;

        // Insert entries using max BATCH_SIZE
        if (batch_index == BATCH_SIZE) {
            h->find_batch(out_batch, BATCH_SIZE);
            cudaCheckErrors();

            checkHybridBatchedOutput(test_entries, out_batch, BATCH_SIZE);
            batch_index = 0;
        }
    }
    // Check remaining entries
    if (batch_index > 0 && batch_index < BATCH_SIZE) {
        h->find_batch(out_batch, batch_index);
        cudaCheckErrors();

        checkHybridBatchedOutput(test_entries, out_batch, batch_index);
    }
    time_t t1_find = get_time_us();
    std::cout << "Hybrid Find time: " << t1_find - t0_find << " us" << std::endl;

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

    test_hybrid_table(test_entries);
    test_cpu_table(test_entries);

    return 0;
}

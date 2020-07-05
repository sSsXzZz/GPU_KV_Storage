#include <chrono>
#include <random>
#include <unordered_map>

#include "hash.h"

using UniformDistribution = std::uniform_int_distribution<uint>;
using Generator = std::mt19937;
using DataMap = std::unordered_map<std::string, std::string>;
using CLOCK = std::chrono::high_resolution_clock;

static constexpr char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static constexpr uint NUM_TEST_ENTRIES = 100000;

void checkEntryEqual(GpuHashEntry& in, GpuHashEntry& out) {
    if (in == out) {
        // std::cout << "Entries " << in << " and " << out << " are equal" << std::endl;
    } else {
        std::cout << "Entries " << in << " and " << out << " are NOT equal!" << std::endl;
        std::abort();
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
        const char* out_key = out_batch->at(i).key;
        const char* out_word = out_batch->at(i).word;

        std::string key(out_batch->at(i).key);
        std::string word(out_batch->at(i).word);
        if (strcmp(key.c_str(), out_key) == 0 && strcmp(word.c_str(), out_word) == 0) {
            // printf("entry(%s, %s) == map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
        } else {
            printf("entry(%s, %s) != map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
            std::abort();
        }
    }
}

void checkCpuBatchedOutput(DataMap& test_data, CpuHashEntryBatch* out_batch, uint num_entries) {
    for (uint i = 0; i < num_entries; i++) {
        const char* out_key = out_batch->entries[i].key;
        const char* out_word = out_batch->entries[i].word;

        std::string key(out_batch->entries[i].key);
        std::string word(out_batch->entries[i].word);
        if (strcmp(key.c_str(), out_key) == 0 && strcmp(word.c_str(), out_word) == 0) {
            // printf("entry(%s, %s) == map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
        } else {
            printf("entry(%s, %s) != map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
            std::abort();
        }
    }
}

void test_cpu_table(DataMap& test_entries) {
    CpuHashTable* h = new CpuHashTable;

    CpuHashEntryBatch* in_batch = new CpuHashEntryBatch;
    CpuHashEntryBatch* out_batch = new CpuHashEntryBatch;

    // Iterate through map and batch insert entries
    auto t0_insert = CLOCK::now();
    int batch_index = 0;  // Count of entries in this batch
    for (auto entry : test_entries) {
        const std::string& key = entry.first;
        const std::string& word = entry.second;
        std::memcpy(&in_batch->entries[batch_index].key, key.c_str(), key.size());
        std::memcpy(&in_batch->entries[batch_index].word, word.c_str(), word.size());
        batch_index++;

        // Insert entries using max CPU_BATCH_SIZE
        if (batch_index == CPU_BATCH_SIZE) {
            h->insert_batch(in_batch, CPU_BATCH_SIZE);
            batch_index = 0;
        }
    }
    // Insert remaining entries
    if (batch_index > 0 && batch_index < CPU_BATCH_SIZE) {
        h->insert_batch(in_batch, CPU_BATCH_SIZE);
    }
    auto t1_insert = CLOCK::now();
    auto tdiff_insert = std::chrono::duration_cast<std::chrono::microseconds>(t1_insert - t0_insert);
    std::cout << "CPU Insert time: " << tdiff_insert.count() / 1000 << "." << tdiff_insert.count() % 1000 << " ms"
              << std::endl;

    // Iterate through map & batch find entries
    auto t0_find = CLOCK::now();
    batch_index = 0;
    for (auto entry : test_entries) {
        const std::string& key = entry.first;
        std::memcpy(&out_batch->entries[batch_index].key, key.c_str(), key.size());
        std::memset(&out_batch->entries[batch_index].word, 0, WORD_SIZE);
        batch_index++;

        // Insert entries using max CPU_BATCH_SIZE
        if (batch_index == CPU_BATCH_SIZE) {
            h->find_batch(out_batch, CPU_BATCH_SIZE);

            checkCpuBatchedOutput(test_entries, out_batch, CPU_BATCH_SIZE);
            batch_index = 0;
        }
    }
    // Check remaining entries
    if (batch_index > 0 && batch_index < CPU_BATCH_SIZE) {
        h->find_batch(out_batch, CPU_BATCH_SIZE);

        checkCpuBatchedOutput(test_entries, out_batch, CPU_BATCH_SIZE);
    }
    auto t1_find = CLOCK::now();
    auto tdiff_find = std::chrono::duration_cast<std::chrono::microseconds>(t1_find - t0_find);
    std::cout << "CPU Find time: " << tdiff_find.count() / 1000 << "." << tdiff_find.count() % 1000 << " ms"
              << std::endl;

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
    auto t0_insert = CLOCK::now();
    int batch_index = 0;  // Count of entries in this batch
    for (auto entry : test_entries) {
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
    auto t1_insert = CLOCK::now();
    auto tdiff_insert = std::chrono::duration_cast<std::chrono::microseconds>(t1_insert - t0_insert);
    std::cout << "GPU Insert time: " << tdiff_insert.count() / 1000 << "." << tdiff_insert.count() % 1000 << " ms"
              << std::endl;

    // Iterate through map & batch find entries
    auto t0_find = CLOCK::now();
    batch_index = 0;
    for (auto entry : test_entries) {
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

        checkGpuBatchedOutput(test_entries, out_batch, BATCH_SIZE);
    }
    auto t1_find = CLOCK::now();
    auto tdiff_find = std::chrono::duration_cast<std::chrono::microseconds>(t1_find - t0_find);
    std::cout << "GPU Find time: " << tdiff_find.count() / 1000 << "." << tdiff_find.count() % 1000 << " ms"
              << std::endl;

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

    /*
    GpuHashTable* h = new GpuHashTable;
    init_hash_table(h);
    cudaCheckErrors();

    GpuHashEntryBatch* in_batch = new GpuHashEntryBatch;
    GpuHashEntryBatch* out_batch = new GpuHashEntryBatch;
    cudaDeviceSynchronize();
    cudaCheckErrors();

    // Iterate through map and batch insert entries
    auto t0_insert = CLOCK::now();
    int batch_index = 0;  // Count of entries in this batch
    for (auto entry : test_entries) {
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
    auto t1_insert = CLOCK::now();
    auto tdiff_insert = std::chrono::duration_cast<std::chrono::microseconds>(t1_insert - t0_insert);
    std::cout << "Insert time: " << tdiff_insert.count() / 1000 << "." << tdiff_insert.count() % 1000 << " ms"
              << std::endl;

    // Iterate through map & batch find entries
    auto t0_find = CLOCK::now();
    batch_index = 0;
    for (auto entry : test_entries) {
        const std::string& key = entry.first;
        std::memcpy(&out_batch->entries[batch_index].key, key.c_str(), key.size());
        std::memset(&out_batch->entries[batch_index].word, 0, WORD_SIZE);
        batch_index++;

        // Insert entries using max BATCH_SIZE
        if (batch_index == BATCH_SIZE) {
            hash_find_batch(h, out_batch, BATCH_SIZE);
            cudaCheckErrors();

            checkBatchedOutput(test_entries, out_batch, BATCH_SIZE);
            batch_index = 0;
        }
    }
    // Check remaining entries
    if (batch_index > 0 && batch_index < BATCH_SIZE) {
        hash_find_batch(h, out_batch, BATCH_SIZE);
        cudaCheckErrors();

        checkBatchedOutput(test_entries, out_batch, BATCH_SIZE);
    }
    auto t1_find = CLOCK::now();
    auto tdiff_find = std::chrono::duration_cast<std::chrono::microseconds>(t1_find - t0_find);
    std::cout << "Find time: " << tdiff_find.count() / 1000 << "." << tdiff_find.count() % 1000 << " ms" << std::endl;

    // print_all_entries(h);

    // std::cout << "_____ UNORDED_MAP ENTRIES _____\n";
    for (auto entry : test_entries) {
        // printf("[%s]=%s\n", entry.first.c_str(), entry.second.c_str());
    }
    // std::cout << "__________________________\n";

    delete h;
    delete in_batch;
    delete out_batch;

    // TODO
    */

    return 0;
}

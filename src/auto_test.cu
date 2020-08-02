#include <chrono>
#include <ctime>
#include <random>
#include <thread>
#include <unordered_map>

#include "cuda_profiler_api.h"
#include "hash.h"
#include "utils.h"

using hash::CpuHashEntry;
using hash::CpuHashEntryBatch;
using hash::CpuHashTable;
using hash::GpuHashTable;
using hash::HybridHashEntryBatch;
using hash::HybridHashTable;

using UniformDistribution = std::uniform_int_distribution<uint>;
using Generator = std::mt19937;
using DataMap = std::unordered_map<std::string, std::string>;

static constexpr char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
static constexpr uint NUM_TEST_ENTRIES = 1000000;

class HashTableTestBase {
  public:
    HashTableTestBase(std::string name) : name_{name} {
    }

    virtual ~HashTableTestBase(){};

    // Inserts every entry in the given DataMap
    virtual void insert_all(DataMap& test_data) = 0;

    // Finds every entry in the given DataMap
    virtual void find_all(DataMap& test_data, bool check_data) = 0;

    // Clears all entries in hash table
    virtual void clear() = 0;

    void test_all(DataMap& test_data, bool check_data) {
        time_t t0_insert = get_time_us();
        insert_all(test_data);
        time_t t1_insert = get_time_us();
        insert_all_times.emplace_back(t1_insert - t0_insert);
        std::cout << name_ << " Insert time: " << t1_insert - t0_insert << " us" << std::endl;

        time_t t0_find = get_time_us();
        find_all(test_data, check_data);
        time_t t1_find = get_time_us();
        find_all_times.emplace_back(t1_find - t0_find);
        std::cout << name_ << " Find time: " << t1_find - t0_find << " us" << std::endl;
    }

    void print_averages() {
        time_t insert_all_avg =
            std::accumulate(insert_all_times.begin(), insert_all_times.end(), 0) / insert_all_times.size();
        time_t find_all_avg = std::accumulate(find_all_times.begin(), find_all_times.end(), 0) / find_all_times.size();

        std::cout << name_ << " Avg Insert time: " << insert_all_avg << " us" << std::endl;
        std::cout << name_ << " Avg Find time: " << find_all_avg << " us" << std::endl;
    }

  protected:
    std::string name_;
    std::vector<time_t> insert_all_times;
    std::vector<time_t> find_all_times;
};

class CpuHashTableTest : public HashTableTestBase {
  public:
    CpuHashTableTest(std::string name) : HashTableTestBase(name) {
        h = new CpuHashTable();
        h->init();

        in_batch = new CpuHashEntryBatch;
        out_batch = new CpuHashEntryBatch;

        in_entry = new CpuHashEntry;
        out_entry = new CpuHashEntry;
    }

    ~CpuHashTableTest() {
        delete h;
        delete in_batch;
        delete out_batch;
    }

    void clear() override {
        h->clear();
    }

  protected:
    void insert_all(DataMap& test_data) override {
        for (auto& entry : test_data) {
            const std::string& key = entry.first;
            const std::string& word = entry.second;

            std::memcpy(&in_entry->key, key.c_str(), key.size());
            std::memcpy(&in_entry->word, word.c_str(), word.size());

            if (in_entry->key[0] == '\0') {
                std::cout << "found the empty key\n";
            }
            h->insert_entry(in_entry);
        }
    }

    void compare_data(DataMap& test_data, CpuHashEntry* out_entry) {
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

    void find_all(DataMap& test_data, bool check_data) override {
        for (auto& entry : test_data) {
            const std::string& key = entry.first;

            std::memcpy(&out_entry->key, key.c_str(), key.size());
            std::memset(&out_entry->word, 0, WORD_SIZE);

            h->find_entry(out_entry);
            if (check_data) {
                compare_data(test_data, out_entry);
            }
        }
    }

    CpuHashTable* h;
    CpuHashEntryBatch* in_batch;
    CpuHashEntryBatch* out_batch;
    CpuHashEntry* in_entry;
    CpuHashEntry* out_entry;
};

class HybridHashTableTest : public HashTableTestBase {
  public:
    HybridHashTableTest(std::string name) : HashTableTestBase(name) {
        h = new HybridHashTable;
        cudaCheckErrors();

        for (uint i = 0; i < NUM_BATCHES; i++) {
            cudaMallocHost(&in_batches[i], sizeof(HybridHashEntryBatch));
        }
        cudaMallocHost(&out_batch, sizeof(HybridHashEntryBatch));
        cudaDeviceSynchronize();
        cudaCheckErrors();
    }

    ~HybridHashTableTest() {
        delete h;
        for (uint i = 0; i < NUM_BATCHES; i++) {
            cudaFreeHost(&in_batches[i]);
        }
        cudaFreeHost(&out_batch);
    }

    void clear() override {
        h->clear();
    }

  protected:
    void insert_all(DataMap& test_data) override {
        uint in_batch_index = 0;
        HybridHashEntryBatch* in_batch = in_batches[in_batch_index];

        uint batch_index = 0;  // Count of entries in this batch
        for (auto& entry : test_data) {
            const std::string& key = entry.first;
            const std::string& word = entry.second;
            std::memcpy(&in_batch->keys[batch_index], key.c_str(), key.size());
            std::memcpy(&in_batch->words[batch_index], word.c_str(), word.size());
            batch_index++;

            // Insert entries using max BATCH_SIZE
            if (batch_index == BATCH_SIZE) {
                h->insert_batch(in_batch, BATCH_SIZE);
                in_batch_index++;
                in_batch = in_batches[in_batch_index];
                // cudaCheckErrors();
                batch_index = 0;
            }
        }
        // Insert remaining entries
        if (batch_index > 0 && batch_index < BATCH_SIZE) {
            h->insert_batch(in_batch, batch_index);
            // cudaCheckErrors();
        }
    }

    void compare_data(DataMap& test_data, HybridHashEntryBatch* out_batch, uint num_entries) {
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

    void find_all(DataMap& test_data, bool check_data) override {
        uint batch_index = 0;
        for (auto& entry : test_data) {
            const std::string& key = entry.first;
            std::memcpy(&out_batch->keys[batch_index], key.c_str(), key.size());
            std::memset(&out_batch->words[batch_index], 0, WORD_SIZE);
            batch_index++;

            // Insert entries using max BATCH_SIZE
            if (batch_index == BATCH_SIZE) {
                h->find_batch(out_batch, BATCH_SIZE);

                if (check_data) {
                    cudaCheckErrors();
                    compare_data(test_data, out_batch, BATCH_SIZE);
                }
                batch_index = 0;
            }
        }
        // Check remaining entries
        if (batch_index > 0 && batch_index < BATCH_SIZE) {
            h->find_batch(out_batch, batch_index);

            if (check_data) {
                cudaCheckErrors();
                compare_data(test_data, out_batch, batch_index);
            }
        }
    }

  protected:
    static constexpr uint NUM_BATCHES = (NUM_TEST_ENTRIES / BATCH_SIZE) + (NUM_TEST_ENTRIES % BATCH_SIZE != 0 ? 1 : 0);

    HybridHashTable* h;
    HybridHashEntryBatch* in_batches[NUM_BATCHES];
    HybridHashEntryBatch* out_batch;
};

std::string get_random_string(UniformDistribution& char_picker, Generator& generator, uint used_space, uint length) {
    // This will make sure our string is already null terminated
    std::string s(length, 0);
    for (uint i = 0; i < used_space; i++) {
        s[i] = alphanum[char_picker(generator)];
    }
    return s;
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

    HybridHashTableTest hybrid_tester("Hybrid");
    cudaProfilerStart();
    hybrid_tester.test_all(test_entries, false);
    cudaProfilerStop();

    CpuHashTableTest cpu_tester("CPU");
    cpu_tester.test_all(test_entries, false);

    return 0;
}

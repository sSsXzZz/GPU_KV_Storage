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
static constexpr uint NUM_TEST_ENTRIES = 1 << 20; // ~1M entries
static constexpr uint NUM_TEST_TIMES = 100;
static constexpr uint NUM_THREADS = 10;
static constexpr bool CHECK_DATA = false;

struct KVPair {
    std::string key;
    std::string word;
};

std::vector<KVPair> data_copy;

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

        std::cout << "---------------------------------------------------\n";
        std::cout << name_ << " Avg Insert time: " << insert_all_avg << " us" << std::endl;
        std::cout << name_ << " Avg Find time: " << find_all_avg << " us" << std::endl;
        std::cout << name_ << " Tested " << insert_all_times.size() << " times" << std::endl;
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
        for (uint i = 0; i < NUM_THREADS; i++) {
            out_entries[i] = new CpuHashEntry;
        }
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
        find_all_mt(test_data, check_data);
        return;

        CpuHashEntry* out_entry = out_entries[0];

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

    void find_all_mt(DataMap& test_data, bool check_data) {
        std::vector<std::thread> threads;
        for (uint i = 0; i < NUM_THREADS; i++) {
            threads.emplace_back([&, index = i]() {
                CpuHashEntry* out_entry = out_entries[index];

                // Test entries are broken up so threads have equal number to process
                uint start_index = (NUM_TEST_ENTRIES / NUM_THREADS) * index;
                uint end_index = (NUM_TEST_ENTRIES / NUM_THREADS) * (index + 1);

                for (uint i = start_index; i < end_index; i++) {
                    const std::string& key = data_copy[i].key;

                    std::memcpy(&out_entry->key, key.c_str(), key.size());
                    std::memset(&out_entry->word, 0, WORD_SIZE);

                    h->find_entry(out_entry);
                    if (check_data) {
                        compare_data(test_data, out_entry);
                    }
                }
            });
        }

        for (std::thread& t : threads) {
            t.join();
        }
    }

    CpuHashTable* h;
    CpuHashEntryBatch* in_batch;
    CpuHashEntryBatch* out_batch;
    CpuHashEntry* in_entry;
    CpuHashEntry* out_entries[NUM_THREADS];
};

class HybridHashTableTest : public HashTableTestBase {
  public:
    HybridHashTableTest(std::string name) : HashTableTestBase(name) {
        h = new HybridHashTable;
        cudaCheckErrors();

        for (uint i = 0; i < NUM_BATCHES; i++) {
            cudaMallocHost(&in_batches[i], sizeof(HybridHashEntryBatch));
            cudaMallocHost(&out_batches[i], sizeof(HybridHashEntryBatch));
        }
        cudaDeviceSynchronize();
        cudaCheckErrors();
    }

    ~HybridHashTableTest() {
        delete h;
        for (uint i = 0; i < NUM_BATCHES; i++) {
            cudaFreeHost(&in_batches[i]);
            cudaFreeHost(&out_batches[i]);
        }
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
                printf("Hex dump of key: ");
                for (uint i = 0; i < KEY_SIZE; i++) {
                    printf("0x%x ", key.c_str()[i]);
                }
                printf("\n");
                abort_with_trace();
            }
            std::string word = test_data[key];

            if (strcmp(key.c_str(), out_key) == 0 && strcmp(word.c_str(), out_word) == 0) {
                // printf("entry(%s, %s) == map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);
            } else {
                printf("GPU: entry(%s, %s) != map_entry(%s, %s)\n", key.c_str(), word.c_str(), out_key, out_word);

                printf("Hex dump: entry(");
                for (uint i = 0; i < KEY_SIZE; i++) {
                    printf("%x", key.c_str()[i]);
                }
                printf(", ");
                for (uint i = 0; i < WORD_SIZE; i++) {
                    printf("%x", word.c_str()[i]);
                }

                printf(") map_entry(");
                for (uint i = 0; i < KEY_SIZE; i++) {
                    printf("%x", out_key[i]);
                }
                printf(", ");
                for (uint i = 0; i < WORD_SIZE; i++) {
                    printf("%x", out_word[i]);
                }
                printf(")\n");

                abort_with_trace();
            }
        }
    }

    void find_all(DataMap& test_data, bool check_data) override {
        find_all_mt(test_data, check_data);
        return;

        uint out_batch_index = 0;
        HybridHashEntryBatch* out_batch = in_batches[out_batch_index];

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

                out_batch_index++;
                out_batch = in_batches[out_batch_index];
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

    /*
        TODO: cleanup the data_copy thing
     */
    void find_all_mt(DataMap& test_data, bool check_data) {
        static_assert(NUM_BATCHES >= NUM_THREADS, "More batches than threads means indexing logic won't work");
        std::vector<std::thread> threads;
        for (uint i = 0; i < NUM_THREADS; i++) {
            threads.emplace_back([&, index = i]() {
                // Test entries are broken up so threads have equal number to process
                uint start_index = (NUM_TEST_ENTRIES / NUM_THREADS) * index;
                uint end_index = (NUM_TEST_ENTRIES / NUM_THREADS) * (index + 1);

                // There should be 1 out_batch per request that will be made
                // Figure out which batches this thread is using
                uint out_batch_index = (NUM_BATCHES / NUM_THREADS) * index;
                HybridHashEntryBatch* out_batch = out_batches[out_batch_index];

                uint batch_index = 0;
                for (uint i = start_index; i < end_index; i++) {
                    const std::string& key = data_copy[i].key;
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

                        out_batch_index++;
                        out_batch = out_batches[out_batch_index];
                    }
                }
                // Insert remaining entries
                if (batch_index > 0 && batch_index < BATCH_SIZE) {
                    h->find_batch(out_batch, batch_index);

                    if (check_data) {
                        cudaCheckErrors();
                        compare_data(test_data, out_batch, batch_index);
                    }
                }
            });
        }

        for (std::thread& t : threads) {
            t.join();
        }
    }

  protected:
    static constexpr uint NUM_BATCHES = (NUM_TEST_ENTRIES / BATCH_SIZE) + (NUM_TEST_ENTRIES % BATCH_SIZE != 0 ? 1 : 0);

    HybridHashTable* h;
    HybridHashEntryBatch* in_batches[NUM_BATCHES];
    HybridHashEntryBatch* out_batches[NUM_BATCHES];
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

    HybridHashTableTest hybrid_tester("Hybrid");
    CpuHashTableTest cpu_tester("CPU");
    for (uint n_test = 0; n_test < NUM_TEST_TIMES; n_test++) {
        data_copy.clear();

        // Generate random key, value pairs
        DataMap test_data;
        while (test_data.size() < NUM_TEST_ENTRIES) {
            std::string key = get_random_string(char_picker, generator, key_size(generator), KEY_SIZE);
            std::string word = get_random_string(char_picker, generator, word_size(generator), WORD_SIZE);

            // std::cout << "Generated entry (" << key << ", " << word << ")" << std::endl;

            test_data[key] = word;
        }

        for (auto& entry : test_data) {
            KVPair kv_pair;
            kv_pair.key = std::string(entry.first);
            kv_pair.word = std::string(entry.second);
            data_copy.emplace_back(std::move(kv_pair));
        }

        std::cout << "Running test " << n_test << "/" << NUM_TEST_TIMES << std::endl;
        cudaProfilerStart();
        hybrid_tester.test_all(test_data, CHECK_DATA);
        cudaProfilerStop();

        cpu_tester.test_all(test_data, CHECK_DATA);

        hybrid_tester.clear();
        cpu_tester.clear();
    }
    hybrid_tester.print_averages();
    cpu_tester.print_averages();

    return 0;
}

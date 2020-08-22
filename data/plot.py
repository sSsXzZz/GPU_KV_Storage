import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_variable_batch():
    # CPU does not batch data, so generate horizontal lines with these values
    CPU_FIND_AVG = 71997
    CPU_FIND_STDDEV = 628
    CPU_MT_FIND_AVG = 13105
    CPU_MT_FIND_STDDEV = 179
    CPU_INSERT_AVG = 119134
    CPU_INSERT_STDDEV = 619
    CPU_MT_INSERT_AVG = 120137
    CPU_MT_INSERT_STDDEV = 941

    batch_data = np.genfromtxt('./csv_files/batch.csv', delimiter=',', names=True, skip_header=1)
    batch_data_mt = np.genfromtxt('./csv_files/batch_mt.csv', delimiter=',', names=True, skip_header=1)
    n_batch_data = len(batch_data['BatchSize'])

    # FIND - Average data
    # --------------------------
    fig_avg_find, ax_avg_find = plt.subplots()

    ax_avg_find.plot(batch_data['BatchSize'], batch_data['HybridFindAvg'], '-o', label='Hybrid Single-Threaded')
    ax_avg_find.plot(batch_data_mt['BatchSize'], batch_data_mt['HybridFindAvg'], '-v', label='Hybrid Multi-Threaded')
    ax_avg_find.plot(batch_data['BatchSize'], np.repeat(CPU_FIND_AVG, n_batch_data), '--s', label='CPU Single-Threaded')
    ax_avg_find.plot(batch_data['BatchSize'], np.repeat(CPU_MT_FIND_AVG, n_batch_data), '--*', label='CPU Multi-Threaded')
    ax_avg_find.legend()
    ax_avg_find.set(title='Find - Average Latency', xlabel='Batch Size', ylabel='Time (us)')
    ax_avg_find.set_xscale('log', base=2)
    ax_avg_find.set_yscale('log', base=10)
    ax_avg_find.grid()

    fig_avg_find.savefig("graphs/batch_avg_find.png", bbox_inches="tight")

    # Insert - Average data
    # --------------------------
    fig_avg_insert, ax_avg_insert = plt.subplots()

    ax_avg_insert.plot(batch_data['BatchSize'], batch_data['HybridInsertAvg'], '-o', label='Hybrid Single-Threaded')
    ax_avg_insert.plot(batch_data_mt['BatchSize'], batch_data_mt['HybridInsertAvg'], '-v', label='Hybrid Multi-Threaded')
    ax_avg_insert.plot(batch_data['BatchSize'], np.repeat(CPU_INSERT_AVG, n_batch_data), '--s', label='CPU Single-Threaded')
    ax_avg_insert.plot(batch_data['BatchSize'], np.repeat(CPU_MT_INSERT_AVG, n_batch_data), '--*', label='CPU Multi-Threaded')
    ax_avg_insert.legend()
    ax_avg_insert.set(title='Insert - Average Latency', xlabel='Batch Size', ylabel='Time (us)')
    ax_avg_insert.set_xscale('log', base=2)
    #ax_avg_insert.set_yscale('log', base=10)
    ax_avg_insert.grid()

    fig_avg_insert.savefig("graphs/batch_avg_insert.png", bbox_inches="tight")

    # FIND - Std Dev Data
    # ---------------------------------
    fig_dev_find, ax_dev_find = plt.subplots()

    ax_dev_find.plot(batch_data['BatchSize'], batch_data['HybridFindStdDev'], '-o', label='Hybrid Single-Threaded')
    ax_dev_find.plot(batch_data_mt['BatchSize'], batch_data_mt['HybridFindStdDev'], '-v', label='Hybrid Multi-Threaded')
    ax_dev_find.plot(batch_data['BatchSize'], np.repeat(CPU_FIND_STDDEV, n_batch_data), '--s', label='CPU Single-Threaded')
    ax_dev_find.plot(batch_data['BatchSize'], np.repeat(CPU_MT_FIND_STDDEV, n_batch_data), '--*', label='CPU Multi-Threaded')
    ax_dev_find.legend()
    ax_dev_find.set(title='Find - Standard Deviation', xlabel='Batch Size', ylabel='Time (us)')
    ax_dev_find.set_xscale('log', base=2)
    #ax_dev_find.set_yscale('log', base=10)
    ax_dev_find.grid()

    fig_dev_find.savefig("graphs/batch_dev_find.png", bbox_inches="tight")

    # Insert - Std Dev Data
    # ---------------------------------
    fig_dev_insert, ax_dev_insert = plt.subplots()

    ax_dev_insert.plot(batch_data['BatchSize'], batch_data['HybridInsertStdDev'], '-o', label='Hybrid Single-Threaded')
    ax_dev_insert.plot(batch_data_mt['BatchSize'], batch_data_mt['HybridInsertStdDev'], '-v', label='Hybrid Multi-Threaded')
    ax_dev_insert.plot(batch_data['BatchSize'], np.repeat(CPU_INSERT_STDDEV, n_batch_data), '--s', label='CPU Single-Threaded')
    ax_dev_insert.plot(batch_data['BatchSize'], np.repeat(CPU_MT_INSERT_STDDEV, n_batch_data), '--*', label='CPU Multi-Threaded')
    ax_dev_insert.legend()
    ax_dev_insert.set(title='Insert - Standard Deviation', xlabel='Batch Size', ylabel='Time (us)')
    ax_dev_insert.set_xscale('log', base=2)
    #ax_dev_insert.set_yscale('log', base=10)
    ax_dev_insert.grid()

    fig_dev_insert.savefig("graphs/batch_dev_insert.png", bbox_inches="tight")

    plt.show()
    
def plot_key():
    key_data = np.genfromtxt('./csv_files/key.csv', delimiter=',', names=True, skip_header=1)
    key_data_mt = np.genfromtxt('./csv_files/key_mt.csv', delimiter=',', names=True, skip_header=1)
    n_key_data = len(key_data['KeySize'])

    # FIND - Average data
    # --------------------------
    fig_avg_find, ax_avg_find = plt.subplots()

    ax_avg_find.plot(key_data['KeySize'], key_data['HybridFindAvg'], '-o', label='Hybrid Single-Threaded')
    ax_avg_find.plot(key_data_mt['KeySize'], key_data_mt['HybridFindAvg'], '-v', label='Hybrid Multi-Threaded')
    ax_avg_find.plot(key_data['KeySize'], key_data['CPUFindAvg'], '--s', label='CPU Single-Threaded')
    ax_avg_find.plot(key_data['KeySize'], key_data_mt['CPUFindAvg'], '--*', label='CPU Multi-Threaded')
    ax_avg_find.legend()
    ax_avg_find.set(title='Find - Average Latency', xlabel='Key Size', ylabel='Time (us)')
    ax_avg_find.set_xscale('log', base=2)
    ax_avg_find.set_yscale('log', base=10)
    ax_avg_find.grid()

    fig_avg_find.savefig("graphs/key_avg_find.png", bbox_inches="tight")

    # Insert - Average data
    # --------------------------
    fig_avg_insert, ax_avg_insert = plt.subplots()

    ax_avg_insert.plot(key_data['KeySize'], key_data['HybridInsertAvg'], '-o', label='Hybrid Single-Threaded')
    ax_avg_insert.plot(key_data_mt['KeySize'], key_data_mt['HybridInsertAvg'], '-v', label='Hybrid Multi-Threaded')
    ax_avg_insert.plot(key_data['KeySize'], key_data['CPUInsertAvg'], '--s', label='CPU Single-Threaded')
    ax_avg_insert.plot(key_data['KeySize'], key_data_mt['CPUInsertAvg'], '--*', label='CPU Multi-Threaded')
    ax_avg_insert.legend()
    ax_avg_insert.set(title='Insert - Average Latency', xlabel='Key Size', ylabel='Time (us)')
    ax_avg_insert.set_xscale('log', base=2)
    ax_avg_insert.set_yscale('log', base=10)
    ax_avg_insert.grid()

    fig_avg_insert.savefig("graphs/key_avg_insert.png", bbox_inches="tight")

    # FIND - Std Dev Data
    # ---------------------------------
    fig_dev_find, ax_dev_find = plt.subplots()

    ax_dev_find.plot(key_data['KeySize'], key_data['HybridFindStdDev'], '-o', label='Hybrid Single-Threaded')
    ax_dev_find.plot(key_data_mt['KeySize'], key_data_mt['HybridFindStdDev'], '-v', label='Hybrid Multi-Threaded')
    ax_dev_find.plot(key_data['KeySize'], key_data['CPUFindStdDev'], '--s', label='CPU Single-Threaded')
    ax_dev_find.plot(key_data['KeySize'], key_data_mt['CPUFindStdDev'], '--*', label='CPU Multi-Threaded')
    ax_dev_find.legend()
    ax_dev_find.set(title='Find - Standard Deviation', xlabel='Key Size', ylabel='Time (us)')
    ax_dev_find.set_xscale('log', base=2)
    #ax_dev_find.set_yscale('log', base=10)
    ax_dev_find.grid()

    fig_dev_find.savefig("graphs/key_dev_find.png", bbox_inches="tight")

    # Insert - Std Dev Data
    # ---------------------------------
    fig_dev_insert, ax_dev_insert = plt.subplots()

    ax_dev_insert.plot(key_data['KeySize'], key_data['HybridInsertStdDev'], '-o', label='Hybrid Single-Threaded')
    ax_dev_insert.plot(key_data_mt['KeySize'], key_data_mt['HybridInsertStdDev'], '-v', label='Hybrid Multi-Threaded')
    ax_dev_insert.plot(key_data['KeySize'], key_data['CPUInsertStdDev'], '--s', label='CPU Single-Threaded')
    ax_dev_insert.plot(key_data['KeySize'], key_data_mt['CPUInsertStdDev'], '--*', label='CPU Multi-Threaded')
    ax_dev_insert.legend()
    ax_dev_insert.set(title='Insert - Standard Deviation', xlabel='Key Size', ylabel='Time (us)')
    ax_dev_insert.set_xscale('log', base=2)
    #ax_dev_insert.set_yscale('log', base=10)
    ax_dev_insert.grid()

    fig_dev_insert.savefig("graphs/key_dev_insert.png", bbox_inches="tight")

    plt.show()
    
def plot_word():
    word_data = np.genfromtxt('./csv_files/word.csv', delimiter=',', names=True, skip_header=1)
    word_data_mt = np.genfromtxt('./csv_files/word_mt.csv', delimiter=',', names=True, skip_header=1)
    n_word_data = len(word_data['WordSize'])

    # FIND - Average data
    # --------------------------
    fig_avg_find, ax_avg_find = plt.subplots()

    ax_avg_find.plot(word_data['WordSize'], word_data['HybridFindAvg'], '-o', label='Hybrid Single-Threaded')
    ax_avg_find.plot(word_data_mt['WordSize'], word_data_mt['HybridFindAvg'], '-v', label='Hybrid Multi-Threaded')
    ax_avg_find.plot(word_data['WordSize'], word_data['CPUFindAvg'], '--s', label='CPU Single-Threaded')
    ax_avg_find.plot(word_data['WordSize'], word_data_mt['CPUFindAvg'], '--*', label='CPU Multi-Threaded')
    ax_avg_find.legend()
    ax_avg_find.set(title='Find - Average Latency', xlabel='Word Size', ylabel='Time (us)')
    ax_avg_find.set_xscale('log', base=2)
    ax_avg_find.set_yscale('log', base=10)
    ax_avg_find.grid()

    fig_avg_find.savefig("graphs/word_avg_find.png", bbox_inches="tight")

    # Insert - Average data
    # --------------------------
    fig_avg_insert, ax_avg_insert = plt.subplots()

    ax_avg_insert.plot(word_data['WordSize'], word_data['HybridInsertAvg'], '-o', label='Hybrid Single-Threaded')
    ax_avg_insert.plot(word_data_mt['WordSize'], word_data_mt['HybridInsertAvg'], '-v', label='Hybrid Multi-Threaded')
    ax_avg_insert.plot(word_data['WordSize'], word_data['CPUInsertAvg'], '--s', label='CPU Single-Threaded')
    ax_avg_insert.plot(word_data['WordSize'], word_data_mt['CPUInsertAvg'], '--*', label='CPU Multi-Threaded')
    ax_avg_insert.legend()
    ax_avg_insert.set(title='Insert - Average Latency', xlabel='Word Size', ylabel='Time (us)')
    ax_avg_insert.set_xscale('log', base=2)
    #ax_avg_insert.set_yscale('log', base=10)
    ax_avg_insert.grid()

    fig_avg_insert.savefig("graphs/word_avg_insert.png", bbox_inches="tight")

    # FIND - Std Dev Data
    # ---------------------------------
    fig_dev_find, ax_dev_find = plt.subplots()

    ax_dev_find.plot(word_data['WordSize'], word_data['HybridFindStdDev'], '-o', label='Hybrid Single-Threaded')
    ax_dev_find.plot(word_data_mt['WordSize'], word_data_mt['HybridFindStdDev'], '-v', label='Hybrid Multi-Threaded')
    ax_dev_find.plot(word_data['WordSize'], word_data['CPUFindStdDev'], '--s', label='CPU Single-Threaded')
    ax_dev_find.plot(word_data['WordSize'], word_data_mt['CPUFindStdDev'], '--*', label='CPU Multi-Threaded')
    ax_dev_find.legend()
    ax_dev_find.set(title='Find - Standard Deviation', xlabel='Word Size', ylabel='Time (us)')
    ax_dev_find.set_xscale('log', base=2)
    #ax_dev_find.set_yscale('log', base=10)
    ax_dev_find.grid()

    fig_dev_find.savefig("graphs/word_dev_find.png", bbox_inches="tight")

    # Insert - Std Dev Data
    # ---------------------------------
    fig_dev_insert, ax_dev_insert = plt.subplots()

    ax_dev_insert.plot(word_data['WordSize'], word_data['HybridInsertStdDev'], '-o', label='Hybrid Single-Threaded')
    ax_dev_insert.plot(word_data_mt['WordSize'], word_data_mt['HybridInsertStdDev'], '-v', label='Hybrid Multi-Threaded')
    ax_dev_insert.plot(word_data['WordSize'], word_data['CPUInsertStdDev'], '--s', label='CPU Single-Threaded')
    ax_dev_insert.plot(word_data['WordSize'], word_data_mt['CPUInsertStdDev'], '--*', label='CPU Multi-Threaded')
    ax_dev_insert.legend()
    ax_dev_insert.set(title='Insert - Standard Deviation', xlabel='Word Size', ylabel='Time (us)')
    ax_dev_insert.set_xscale('log', base=2)
    #ax_dev_insert.set_yscale('log', base=10)
    ax_dev_insert.grid()

    fig_dev_insert.savefig("graphs/word_dev_insert.png", bbox_inches="tight")

    plt.show()

def plot_thread():
    thread_data = np.genfromtxt('./csv_files/thread.csv', delimiter=',', names=True, skip_header=1)
    n_thread_data = len(thread_data['Threads'])

    # FIND - Average data
    # --------------------------
    fig_avg_find, ax_avg_find = plt.subplots()

    ax_avg_find.plot(thread_data['Threads'], thread_data['HybridFindAvg'], '-o', label='Hybrid Find')
    ax_avg_find.plot(thread_data['Threads'], thread_data['CPUFindAvg'], '--s', label='CPU Find')
    ax_avg_find.legend()
    ax_avg_find.set(title='Find - Average Latency', xlabel='Threads', ylabel='Time (us)')
    ax_avg_find.set_xscale('log', base=2)
    ax_avg_find.set_yscale('log', base=10)
    ax_avg_find.grid()

    fig_avg_find.savefig("graphs/thread_avg_find.png", bbox_inches="tight")

    # Insert - Average data
    # --------------------------
    fig_avg_insert, ax_avg_insert = plt.subplots()

    ax_avg_insert.plot(thread_data['Threads'], thread_data['HybridInsertAvg'], '-o', label='Hybrid Insert')
    ax_avg_insert.plot(thread_data['Threads'], thread_data['CPUInsertAvg'], '--s', label='CPU Insert')
    ax_avg_insert.legend()
    ax_avg_insert.set(title='Insert - Average Latency', xlabel='Threads', ylabel='Time (us)')
    ax_avg_insert.set_xscale('log', base=2)
    #ax_avg_insert.set_yscale('log', base=10)
    ax_avg_insert.grid()

    fig_avg_insert.savefig("graphs/thread_avg_insert.png", bbox_inches="tight")

    # FIND - Std Dev Data
    # ---------------------------------
    fig_dev_find, ax_dev_find = plt.subplots()

    ax_dev_find.plot(thread_data['Threads'], thread_data['HybridFindStdDev'], '-o', label='Hybrid Find')
    ax_dev_find.plot(thread_data['Threads'], thread_data['CPUFindStdDev'], '--s', label='CPU Find')
    ax_dev_find.legend()
    ax_dev_find.set(title='Find - Standard Deviation', xlabel='Threads', ylabel='Time (us)')
    ax_dev_find.set_xscale('log', base=2)
    #ax_dev_find.set_yscale('log', base=10)
    ax_dev_find.grid()

    fig_dev_find.savefig("graphs/thread_dev_find.png", bbox_inches="tight")

    # Insert - Std Dev Data
    # ---------------------------------
    fig_dev_insert, ax_dev_insert = plt.subplots()

    ax_dev_insert.plot(thread_data['Threads'], thread_data['HybridInsertStdDev'], '-o', label='Hybrid Insert')
    ax_dev_insert.plot(thread_data['Threads'], thread_data['CPUInsertStdDev'], '--s', label='CPU Insert')
    ax_dev_insert.legend()
    ax_dev_insert.set(title='Insert - Standard Deviation', xlabel='Threads', ylabel='Time (us)')
    ax_dev_insert.set_xscale('log', base=2)
    #ax_dev_insert.set_yscale('log', base=10)
    ax_dev_insert.grid()

    fig_dev_insert.savefig("graphs/thread_dev_insert.png", bbox_inches="tight")

    plt.show()

# plot_variable_batch()
plot_key()
# plot_word()
# plot_thread()

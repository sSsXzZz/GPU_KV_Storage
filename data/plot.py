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

    batch_data = np.genfromtxt('./csv_files/batch.csv', delimiter=',', names=True, skip_header=1)
    batch_data_mt = np.genfromtxt('./csv_files/batch_mt.csv', delimiter=',', names=True, skip_header=1)
    n_batch_data = len(batch_data['BatchSize'])

    # Average data
    # --------------------------
    fig_avg_find, ax_avg_find = plt.subplots()

    ax_avg_find.plot(batch_data['BatchSize'], batch_data['HybridFindAvg'], '-o', label='Hybrid Single-threaded Find')
    ax_avg_find.plot(batch_data_mt['BatchSize'], batch_data_mt['HybridFindAvg'], '-v', label='Hybrid Multi-threaded Find')
    ax_avg_find.plot(batch_data['BatchSize'], np.repeat(CPU_FIND_AVG, n_batch_data), '--s', label='CPU Single-threaded Find')
    ax_avg_find.plot(batch_data['BatchSize'], np.repeat(CPU_MT_FIND_AVG, n_batch_data), '--*', label='CPU Multi-threaded Find')
    ax_avg_find.legend()
    ax_avg_find.set(title='Find Average Times', xlabel='Batch Size', ylabel='Time (us)')
    ax_avg_find.set_xscale('log', base=2)
    ax_avg_find.set_yscale('log', base=10)
    ax_avg_find.grid()

    fig_avg_find.savefig("graphs/batch_avg_find.png")

    # Std Dev Data
    # ---------------------------------
    fig_dev_find, ax_dev_find = plt.subplots()

    ax_dev_find.plot(batch_data['BatchSize'], batch_data['HybridFindStdDev'], '-o', label='Hybrid Single-threaded Find')
    ax_dev_find.plot(batch_data_mt['BatchSize'], batch_data_mt['HybridFindStdDev'], '-v', label='Hybrid Multi-threaded Find')
    ax_dev_find.plot(batch_data['BatchSize'], np.repeat(CPU_FIND_STDDEV, n_batch_data), '--s', label='CPU Single-threaded Find')
    ax_dev_find.plot(batch_data['BatchSize'], np.repeat(CPU_MT_FIND_STDDEV, n_batch_data), '--*', label='CPU Multi-threaded Find')
    ax_dev_find.legend()
    ax_dev_find.set(title='Find Standard Deviations', xlabel='Batch Size', ylabel='Time (us)')
    ax_dev_find.set_xscale('log', base=2)
    #ax_dev_find.set_yscale('log', base=10)
    ax_dev_find.grid()

    fig_dev_find.savefig("graphs/batch_dev_find.png")

    plt.show()
    
def plot_key():
    key_data = np.genfromtxt('./csv_files/key.csv', delimiter=',', names=True, skip_header=1)
    key_data_mt = np.genfromtxt('./csv_files/key_mt.csv', delimiter=',', names=True, skip_header=1)
    n_key_data = len(key_data['KeySize'])

    # Average data
    # --------------------------
    fig_avg_find, ax_avg_find = plt.subplots()

    ax_avg_find.plot(key_data['KeySize'], key_data['HybridFindAvg'], '-o', label='Hybrid Single-threaded Find')
    ax_avg_find.plot(key_data_mt['KeySize'], key_data_mt['HybridFindAvg'], '-v', label='Hybrid Multi-threaded Find')
    ax_avg_find.plot(key_data['KeySize'], key_data['CPUFindAvg'], '--s', label='CPU Single-threaded Find')
    ax_avg_find.plot(key_data['KeySize'], key_data_mt['CPUFindAvg'], '--*', label='CPU Multi-threaded Find')
    ax_avg_find.legend()
    ax_avg_find.set(title='Find Average Times', xlabel='Key Size', ylabel='Time (us)')
    ax_avg_find.set_xscale('log', base=2)
    ax_avg_find.set_yscale('log', base=10)
    ax_avg_find.grid()

    fig_avg_find.savefig("graphs/key_avg_find.png")

    # Std Dev Data
    # ---------------------------------
    fig_dev_find, ax_dev_find = plt.subplots()

    ax_dev_find.plot(key_data['KeySize'], key_data['HybridFindStdDev'], '-o', label='Hybrid Single-threaded Find')
    ax_dev_find.plot(key_data_mt['KeySize'], key_data_mt['HybridFindStdDev'], '-v', label='Hybrid Multi-threaded Find')
    ax_dev_find.plot(key_data['KeySize'], key_data['CPUFindStdDev'], '--s', label='CPU Single-threaded Find')
    ax_dev_find.plot(key_data['KeySize'], key_data_mt['CPUFindStdDev'], '--*', label='CPU Multi-threaded Find')
    ax_dev_find.legend()
    ax_dev_find.set(title='Find Standard Deviations', xlabel='Key Size', ylabel='Time (us)')
    ax_dev_find.set_xscale('log', base=2)
    #ax_dev_find.set_yscale('log', base=10)
    ax_dev_find.grid()

    fig_dev_find.savefig("graphs/key_dev_find.png")

    plt.show()


# plot_variable_batch()
plot_key()

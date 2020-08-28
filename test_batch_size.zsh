for POWER in `seq 4 12`; do
    NUM_THREADS=$(echo "2^${POWER}" | bc)
    rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
    echo "--------------------------------------------"
    echo "NUM_THREADS ${NUM_THREADS}"
    cmake -E env "CUDAFLAGS=-DNUM_THREADS=\"${NUM_THREADS}\"" cmake .
    make
    ./hash
done

# Powers from 16 (2^4) -> 4096 (2^12)

for POWER in `seq 2 6`; do
    NUM_THREADS=$(echo "2^${POWER}" | bc)
    rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
    echo "--------------------------------------------"
    echo "NUM_THREADS ${NUM_THREADS}"
    cmake -E env "CUDAFLAGS=-DNUM_THREADS=\"${NUM_THREADS}\"" cmake .
    make
    ./hash
done

# Powers from 2 (2^2) -> 64 (2^6)

for POWER in `seq 5 9`; do
    KEY_SIZE=$(echo "2^${POWER}-17" | bc)
    rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
    echo "--------------------------------------------"
    echo "KEY_SIZE ${KEY_SIZE}"
    cmake -E env "CUDAFLAGS=-DKEY_SIZE=\"${KEY_SIZE}\"" cmake .
    make
    ./hash
done

# Powers from 32 (2^5) -> 512 (2^9)
# Key size is value - 17 (to account for word_size of 16 + 1 byte for metadata)

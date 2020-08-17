for POWER in `seq 5 9`; do
    WORD_SIZE=$(echo "2^${POWER}-16" | bc)
    rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
    echo "--------------------------------------------"
    echo "WORD_SIZE ${WORD_SIZE}"
    cmake -E env "CUDAFLAGS=-DWORD_SIZE=\"${WORD_SIZE}\"" cmake .
    make
    ./hash
done

# Powers from 32 (2^5) -> 512 (2^9)
# Word size is value - 16 (to account for key_size of 15 + 1 byte for metadata)

for ENTRIES in `seq 209714 104857 2097512`; do
    rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
    echo "--------------------------------------------"
    echo "Entries ${ENTRIES}"
    cmake -E env "CUDAFLAGS=-DNUM_TEST_ENTRIES=\"(${ENTRIES})\"" cmake .
    make
    ./hash
done

# 104857 104857 2097512 for non-mt
# 209714 104857 2097512 for mt

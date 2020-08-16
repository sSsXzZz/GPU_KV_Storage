for POWER in `seq 4 20`; do
    rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
    echo "--------------------------------------------"
    echo "Power ${POWER}"
    cmake -E env "CUDAFLAGS=-DBATCH_SIZE=\"(1 << ${POWER})\"" cmake .
    make
    ./hash
done

# 4...20 for non-mt
# 4...15 for mt

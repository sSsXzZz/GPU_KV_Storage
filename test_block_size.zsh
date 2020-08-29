for MULTI_THREADED in `seq 0 1`; do
    USE_MULTITHREADED="false"
    if [ $MULTI_THREADED -eq "1" ]
    then
        USE_MULTITHREADED="true"
    fi

    for POWER in `seq 0 12`; do
        BLOCK_SIZE=$(echo "2^${POWER}" | bc)
        rm -rf CMakeFiles/ compile_commands.json CMakeCache.txt cmake_install.cmake
        echo "--------------------------------------------"
        echo "BLOCK_SIZE ${BLOCK_SIZE} MULTI_THREADED ${USE_MULTITHREADED}"
        cmake -E env "CUDAFLAGS=-DBLOCK_SIZE=\"${BLOCK_SIZE}\" -DUSE_MULTITHREADED=\"${USE_MULTITHREADED}\"" cmake .
        make
        ./hash
    done
done

# Powers from 1 (2^0) -> 4096 (2^12)

cmake_minimum_required(VERSION 3.10)

project(hash CXX CUDA)

file(GLOB SOURCE_FILES src/*)

# For YouCompleteMe
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setting optimizations to 0 for debugging
set(CMAKE_CUDA_FLAGS
    ${CMAKE_CUDA_FLAGS};
    "-g -G -O0 -gencode arch=compute_61,code=sm_61"
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)


cmake_minimum_required(VERSION 3.10)

project(hash CXX CUDA)

file(GLOB SOURCE_FILES src/*)

# Setting optimizations to 0 for debugging
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_61,code=sm_61"
    )
# NOTE: Lower optimization for debugging
set(DEBUG_FLAGS "-g -G")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DEBUG_FLAGS}")

message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# For YouCompleteMe
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)


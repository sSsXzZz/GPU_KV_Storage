cmake_minimum_required(VERSION 3.10)

project(hash CXX CUDA)

file(GLOB SOURCE_FILES src/*)

# For YouCompleteMe
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)


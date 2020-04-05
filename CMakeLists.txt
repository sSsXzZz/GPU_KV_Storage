cmake_minimum_required(VERSION 3.10)

project(hash CXX CUDA)

set(SOURCE_FILES hash.h hash.cu)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


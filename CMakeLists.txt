cmake_minimum_required(VERSION 3.12)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)
#find_package(OpenMP REQUIRED)

#project(bitonic LANGUAGES CUDA CXX)
add_executable(matmul matmul.cpp)

#message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")

#include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(matmul PRIVATE caliper)

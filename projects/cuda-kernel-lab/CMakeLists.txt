cmake_minimum_required(VERSION 3.20)

project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_KERNEL_LAB_CUDA_ARCHITECTURES "" CACHE STRING "CUDA architectures to build for, for example 86 or 86;89")

find_package(CUDAToolkit REQUIRED)

if(CUDA_KERNEL_LAB_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_KERNEL_LAB_CUDA_ARCHITECTURES}")
elseif(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(cuda_vector_add src/vector_add.cu)
target_link_libraries(cuda_vector_add PRIVATE CUDA::cudart)

add_executable(cuda_reduce_sum src/reduce_sum.cu)
target_link_libraries(cuda_reduce_sum PRIVATE CUDA::cudart)

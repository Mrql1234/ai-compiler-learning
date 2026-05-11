cmake_minimum_required(VERSION 3.20)

project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CUDA_KERNEL_LAB_ENABLE_NATIVE_ARCH "Build for the native GPU architecture" ON)

find_package(CUDAToolkit REQUIRED)

if(CUDA_KERNEL_LAB_ENABLE_NATIVE_ARCH)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(cuda_vector_add src/vector_add.cu)
target_link_libraries(cuda_vector_add PRIVATE CUDA::cudart)

add_executable(cuda_reduce_sum src/reduce_sum.cu)
target_link_libraries(cuda_reduce_sum PRIVATE CUDA::cudart)


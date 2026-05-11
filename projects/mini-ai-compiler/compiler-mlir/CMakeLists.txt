cmake_minimum_required(VERSION 3.20)
project(mini-compiler-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CUDAToolkit QUIET)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM CMake path: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR CMake path: ${MLIR_CMAKE_DIR}")
if(CUDAToolkit_FOUND)
  message(STATUS "Found CUDA toolkit: ${CUDAToolkit_VERSION}")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA standard to conform to")
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(MINI_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures used by mini compiler CUDA benchmark targets")
  set(MINI_NVTX_LIBRARY "")
  if(TARGET CUDA::nvToolsExt)
    set(MINI_NVTX_LIBRARY CUDA::nvToolsExt)
  elseif(CUDA_nvToolsExt_LIBRARY)
    set(MINI_NVTX_LIBRARY "${CUDA_nvToolsExt_LIBRARY}")
  endif()
  if(MINI_NVTX_LIBRARY)
    message(STATUS "Found NVTX library: ${MINI_NVTX_LIBRARY}")
  else()
    message(WARNING "NVTX library not found; NVTX ranges will be compiled out")
  endif()
else()
  message(WARNING "CUDA toolkit not found; building without MiniCudaRuntimeWrappers")
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(lib)
if(CUDAToolkit_FOUND)
  add_subdirectory(runtime)
endif()
add_subdirectory(tools)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
  add_subdirectory(test)
endif()

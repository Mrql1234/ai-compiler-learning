cmake_minimum_required(VERSION 3.20)
project(mlir-passes LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 LLVM 和 MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM CMake path: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR CMake path: ${MLIR_CMAKE_DIR}")

# 设置包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加编译选项
add_definitions(${LLVM_DEFINITIONS})

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加 Pass 库
add_subdirectory(lib)

# 添加测试
add_subdirectory(test)

# 添加工具
add_subdirectory(tools)

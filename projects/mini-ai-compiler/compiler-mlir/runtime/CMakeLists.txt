add_library(MiniCudaRuntimeWrappers SHARED
  CudaRuntimeWrappers.cpp
)

target_include_directories(MiniCudaRuntimeWrappers
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MiniCudaRuntimeWrappers
  PRIVATE
  CUDA::cuda_driver
)

set_target_properties(MiniCudaRuntimeWrappers PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

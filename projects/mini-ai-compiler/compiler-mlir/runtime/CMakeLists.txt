add_library(MiniCudaRuntimeWrappers SHARED
  CudaRuntimeWrappers.cpp
  MiniCudaKernelRuntime.cu
)

target_include_directories(MiniCudaRuntimeWrappers
  PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MiniCudaRuntimeWrappers
  PRIVATE
  CUDA::cuda_driver
  CUDA::cudart
  CUDA::cublas
)

set_target_properties(MiniCudaRuntimeWrappers PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${MINI_CUDA_ARCHITECTURES}"
)

if(MINI_NVTX_LIBRARY)
  target_compile_definitions(MiniCudaRuntimeWrappers
    PRIVATE
    MINI_ENABLE_NVTX
  )
  target_include_directories(MiniCudaRuntimeWrappers
    PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(MiniCudaRuntimeWrappers
    PRIVATE
    ${MINI_NVTX_LIBRARY}
  )
endif()

set_target_properties(MiniCudaRuntimeWrappers PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

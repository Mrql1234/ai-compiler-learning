add_executable(mini-compiler-opt
  mini-compiler-opt.cpp
)

add_executable(mini-compiler-runner
  mini-compiler-runner.cpp
)

add_executable(mini-compiler-gpu-runner
  mini-compiler-gpu-runner.cpp
)

list(GET LLVM_LIBRARY_DIRS 0 MINI_LLVM_LIB_DIR)

target_include_directories(mini-compiler-opt
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_include_directories(mini-compiler-runner
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_include_directories(mini-compiler-gpu-runner
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(mini-compiler-opt
  PRIVATE
  MiniCompilerMLIR
  MLIRFuncDialect
  MLIRIR
  MLIRMlirOptMain
)

target_link_libraries(mini-compiler-runner
  PRIVATE
  MiniCompilerMLIR
  MLIRExecutionEngine
  MLIRFuncDialect
  MLIRIR
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
)

target_link_libraries(mini-compiler-gpu-runner
  PRIVATE
  MiniCompilerMLIR
  MLIRExecutionEngine
  MLIRFuncDialect
  MLIRIR
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  ${CMAKE_DL_LIBS}
)

target_compile_definitions(mini-compiler-gpu-runner
  PRIVATE
  MINI_MLIR_RUNNER_UTILS_PATH="${MINI_LLVM_LIB_DIR}/libmlir_runner_utils.so"
  MINI_MLIR_C_RUNNER_UTILS_PATH="${MINI_LLVM_LIB_DIR}/libmlir_c_runner_utils.so"
)

if(TARGET MiniCudaRuntimeWrappers)
  target_compile_definitions(mini-compiler-gpu-runner
    PRIVATE
    MINI_CUDA_RUNTIME_WRAPPERS_PATH="$<TARGET_FILE:MiniCudaRuntimeWrappers>"
  )
else()
  target_compile_definitions(mini-compiler-gpu-runner
    PRIVATE
    MINI_CUDA_RUNTIME_WRAPPERS_PATH=""
  )
endif()

if(MINI_NVTX_ENABLED)
  target_compile_definitions(mini-compiler-gpu-runner
    PRIVATE
    MINI_ENABLE_NVTX
  )
  target_include_directories(mini-compiler-gpu-runner
    PRIVATE
    ${MINI_NVTX_INCLUDE_DIR}
  )
  target_link_libraries(mini-compiler-gpu-runner
    PRIVATE
    ${MINI_NVTX_LINK_LIBS}
  )
endif()

if(CUDAToolkit_FOUND)
  add_executable(mini-compiler-kernel-bench
    mini-compiler-kernel-bench.cpp
    KernelBenchCuda.cu
  )

  target_include_directories(mini-compiler-kernel-bench
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLVM_INCLUDE_DIRS}
  )

  target_link_libraries(mini-compiler-kernel-bench
    PRIVATE
    LLVMSupport
    CUDA::cudart
    CUDA::cublas
  )

  set_target_properties(mini-compiler-kernel-bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${MINI_CUDA_ARCHITECTURES}"
  )

  if(MINI_NVTX_ENABLED)
    target_compile_definitions(mini-compiler-kernel-bench
      PRIVATE
      MINI_ENABLE_NVTX
    )
    target_include_directories(mini-compiler-kernel-bench
      PRIVATE
      ${MINI_NVTX_INCLUDE_DIR}
    )
    target_link_libraries(mini-compiler-kernel-bench
      PRIVATE
      ${MINI_NVTX_LINK_LIBS}
    )
  endif()
endif()

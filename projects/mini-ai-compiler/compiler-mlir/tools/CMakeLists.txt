add_executable(mini-compiler-opt
  mini-compiler-opt.cpp
)

add_executable(mini-compiler-runner
  mini-compiler-runner.cpp
)

add_executable(mini-compiler-gpu-runner
  mini-compiler-gpu-runner.cpp
)

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
)

target_compile_definitions(mini-compiler-gpu-runner
  PRIVATE
  MINI_CUDA_RUNTIME_WRAPPERS_PATH="$<TARGET_FILE:MiniCudaRuntimeWrappers>"
  MINI_MLIR_RUNNER_UTILS_PATH="/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/libmlir_runner_utils.so"
  MINI_MLIR_C_RUNNER_UTILS_PATH="/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/libmlir_c_runner_utils.so"
)

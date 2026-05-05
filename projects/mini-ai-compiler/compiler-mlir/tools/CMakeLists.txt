add_executable(mini-compiler-opt
  mini-compiler-opt.cpp
)

add_executable(mini-compiler-runner
  mini-compiler-runner.cpp
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
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
)

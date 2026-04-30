add_executable(mini-compiler-opt
  mini-compiler-opt.cpp
)

target_include_directories(mini-compiler-opt
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

add_executable(mlir-passes-opt
  mlir-passes-opt.cpp
)

target_include_directories(mlir-passes-opt
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(mlir-passes-opt
  PRIVATE
  MLIRPasses
  MLIRFuncDialect
  MLIRIR
  MLIRMlirOptMain
)

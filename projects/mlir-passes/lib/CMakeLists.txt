add_library(MLIRPasses
  ConstantFoldPass.cpp
  ExtendedConstantFoldPass.cpp
  TemplateConstantFoldPass.cpp
  BuiltinCanonicalizeFoldPass.cpp
  PassRegistry.cpp
)

target_include_directories(MLIRPasses
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(MLIRPasses
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithmeticDialect
)

# MLIR Passes 库

add_mlir_library(MLIRPasses
  ConstantFoldPass.cpp
  DeadCodeElimPass.cpp
  OperatorFusionPass.cpp

  DEPENDS
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRArithDialect
  MLIRMathDialect
)

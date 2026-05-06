add_mlir_library(MiniCompilerMLIR
  MiniDialect.cpp
  Passes.cpp

  DEPENDS
  MLIRIR

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRRewrite
  MLIRSCFToGPU
  MLIRTensorDialect
  MLIRTransforms
)

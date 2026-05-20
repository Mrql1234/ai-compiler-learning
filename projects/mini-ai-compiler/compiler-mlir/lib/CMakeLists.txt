add_mlir_library(MiniCompilerMLIR
  MiniDialect.cpp
  GpuPasses.cpp
  Passes.cpp
  QuantPasses.cpp

  DEPENDS
  MLIRIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFToGPU
  MLIRTensorDialect
  MLIRTransforms
)

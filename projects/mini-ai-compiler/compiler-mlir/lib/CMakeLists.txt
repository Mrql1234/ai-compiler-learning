add_mlir_library(MiniCompilerMLIR
  MiniDialect.cpp
  Passes.cpp

  DEPENDS
  MLIRIR

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)

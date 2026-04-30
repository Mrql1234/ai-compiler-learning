#include "MiniCompiler/MiniDialect.h"
#include "MiniCompiler/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mini::MiniDialect>();

  mini::registerMiniPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mini compiler optimizer\n", registry));
}

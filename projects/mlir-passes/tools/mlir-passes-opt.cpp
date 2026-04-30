#include "Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();

  mlir::registerAllCustomPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "mlir-passes external driver\n", registry));
}

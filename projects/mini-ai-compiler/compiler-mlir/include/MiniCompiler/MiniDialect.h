#ifndef MINI_COMPILER_MINI_DIALECT_H
#define MINI_COMPILER_MINI_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace mini {

class MiniDialect : public mlir::Dialect {
public:
  explicit MiniDialect(mlir::MLIRContext *context);

  static llvm::StringRef getDialectNamespace() { return "mini"; }
};

} // namespace mini

#endif // MINI_COMPILER_MINI_DIALECT_H

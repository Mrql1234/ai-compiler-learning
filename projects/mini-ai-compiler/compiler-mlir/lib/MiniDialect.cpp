#include "MiniCompiler/MiniDialect.h"

using namespace mlir;

namespace mini {

MiniDialect::MiniDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniDialect>()) {
  allowUnknownOperations();
}

} // namespace mini

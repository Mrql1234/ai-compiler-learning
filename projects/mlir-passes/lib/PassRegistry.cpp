#include "Passes.h"

namespace mlir {

void registerAllCustomPasses() {
  registerConstantFoldPass();
  registerExtendedConstantFoldPass();
  registerTemplateConstantFoldPass();
  registerBuiltinCanonicalizeFoldPass();
}

} // namespace mlir

from .constant_fold import ConstantFoldPass
from .dce import DCEPass
from .fusion import FusionPass
from .mlir_canonicalize import MLIRCanonicalizePass

__all__ = ["ConstantFoldPass", "DCEPass", "FusionPass", "MLIRCanonicalizePass"]

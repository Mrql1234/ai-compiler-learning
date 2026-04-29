from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from examples.mlp import TinyMLP
from frontend.fx_importer import FXImporter
from ir.mlir_printer import print_mlir
from ir.printer import write_mlir
from passes.constant_fold import ConstantFoldPass
from passes.dce import DCEPass
from passes.fusion import FusionPass
from passes.manager import PassManager


def main() -> None:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    sample = torch.randn(2, 4)
    graph = FXImporter().import_model(model, (sample,))
    optimized = PassManager([ConstantFoldPass(), DCEPass(), FusionPass(), DCEPass()]).run(graph)

    path = write_mlir(optimized, PROJECT_ROOT / "artifacts" / "mlp_optimized.mlir")
    print(print_mlir(optimized))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()

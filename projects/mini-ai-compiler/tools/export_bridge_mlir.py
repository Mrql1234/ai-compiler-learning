from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from examples.mlp import TinyMLP
from frontend.fx_importer import FXImporter
from ir.printer import write_mlir


def main() -> None:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    sample = torch.randn(2, 4)
    graph = FXImporter().import_model(model, (sample,))
    path = write_mlir(
        graph,
        PROJECT_ROOT / "artifacts" / "mlp_bridge_input.mlir",
        generic_ops=True,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()

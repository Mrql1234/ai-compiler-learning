from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from examples.mlp import TinyMLP
from backend.triton.lowering import TritonLowerer
from frontend.fx_importer import FXImporter
from ir.mlir_printer import print_mlir
from ir.printer import write_graph, write_mlir
from passes.constant_fold import ConstantFoldPass
from passes.dce import DCEPass
from passes.fusion import FusionPass
from passes.manager import PassManager


def main() -> None:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    sample = torch.randn(2, 4)

    importer = FXImporter()
    graph = importer.import_model(model, (sample,))
    output_dir = PROJECT_ROOT / "artifacts"
    original_path = write_graph(graph, output_dir / "mlp_original.ir")
    original_mlir = write_mlir(graph, output_dir / "mlp_original.mlir")

    print("=== Original IR ===")
    print(graph)
    print(f"saved: {original_path}")
    print("=== Original MLIR-style IR ===")
    print(print_mlir(graph))
    print(f"saved: {original_mlir}")

    optimized = PassManager([ConstantFoldPass(), DCEPass(), FusionPass(), DCEPass()]).run(graph)
    optimized_path = write_graph(optimized, output_dir / "mlp_optimized.ir")
    optimized_mlir = write_mlir(optimized, output_dir / "mlp_optimized.mlir")
    print("\n=== Optimized IR ===")
    print(optimized)
    print(f"saved: {optimized_path}")
    print("=== Optimized MLIR-style IR ===")
    print(print_mlir(optimized))
    print(f"saved: {optimized_mlir}")

    lowered = TritonLowerer().lower(optimized)
    print("=== Triton Lowering Plan ===")
    for line in lowered.execution_plan:
        print(line)
    if lowered.unsupported_ops:
        print("unsupported:", ", ".join(sorted(set(lowered.unsupported_ops))))


if __name__ == "__main__":
    main()

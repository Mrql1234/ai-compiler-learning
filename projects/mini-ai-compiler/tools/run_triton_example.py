from __future__ import annotations

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from backend.triton.executor import TritonExecutor
from backend.triton.lowering import TritonLowerer
from examples.mlp import TinyMLP
from frontend.fx_importer import FXImporter
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

    lowering = TritonLowerer().lower(optimized)
    print("=== Triton Lowering ===")
    for line in lowering.execution_plan:
        print(line)

    executor = TritonExecutor()
    if not executor.available():
        print("Triton backend unavailable: CUDA runtime not detected.")
        return

    outputs = executor.run(optimized, {"x": sample.detach().cpu().numpy()})
    eager = model(sample).detach().cpu().numpy()
    print("allclose:", np.allclose(outputs[0], eager, atol=1e-5))
    print("triton:", outputs[0])
    print("eager :", eager)


if __name__ == "__main__":
    main()

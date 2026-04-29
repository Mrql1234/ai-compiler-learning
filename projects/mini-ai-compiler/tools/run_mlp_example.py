from __future__ import annotations

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from backend.cpu.executor import CPUExecutor
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

    importer = FXImporter()
    graph = importer.import_model(model, (sample,))
    print("=== Original IR ===")
    print(graph)

    manager = PassManager([ConstantFoldPass(), DCEPass(), FusionPass(), DCEPass()])
    manager.run(graph)

    print("\n=== Optimized IR ===")
    print(graph)

    executor = CPUExecutor()
    outputs = executor.run(graph, {"x": sample.detach().cpu().numpy()})
    eager = model(sample).detach().cpu().numpy()

    print("\n=== Output Check ===")
    print("compiler:", outputs[0])
    print("eager   :", eager)
    print("allclose:", np.allclose(outputs[0], eager, atol=1e-5))


if __name__ == "__main__":
    main()

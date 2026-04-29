from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from backend.cpu.executor import CPUExecutor
from examples.mlp import TinyMLP
from frontend.fx_importer import FXImporter
from passes.constant_fold import ConstantFoldPass
from passes.dce import DCEPass
from passes.fusion import FusionPass
from passes.manager import PassManager


def benchmark(function, repeat: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples)


def main() -> None:
    torch.manual_seed(0)
    model = TinyMLP().eval()
    sample = torch.randn(32, 4)

    importer = FXImporter()
    graph = importer.import_model(model, (sample,))
    optimized = PassManager([ConstantFoldPass(), DCEPass(), FusionPass(), DCEPass()]).run(graph)
    executor = CPUExecutor()
    cpu_inputs = {"x": sample.detach().cpu().numpy()}

    eager_ms = benchmark(lambda: model(sample))
    compiler_ms = benchmark(lambda: executor.run(optimized, cpu_inputs))
    eager_out = model(sample).detach().cpu().numpy()
    compiler_out = executor.run(optimized, cpu_inputs)[0]

    print("=== Benchmark: TinyMLP ===")
    print(f"eager_ms_mean    : {eager_ms:.4f}")
    print(f"compiler_ms_mean : {compiler_ms:.4f}")
    print(f"allclose         : {np.allclose(eager_out, compiler_out, atol=1e-5)}")


if __name__ == "__main__":
    main()

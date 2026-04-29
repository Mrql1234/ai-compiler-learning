from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from backend.triton.lowering import TritonLowerer
from ir.graph import Graph
from ir.node import Node


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

    from backend.triton.executor import TritonExecutor


class TritonLoweringTests(unittest.TestCase):
    def test_lowering_plan_for_fused_linear_relu(self) -> None:
        graph = Graph(name="lower")
        inp = graph.add_input("x")
        weight = graph.add_constant("w", np.array([[1.0, -1.0]], dtype=np.float32))
        bias = graph.add_constant("b", np.array([0.5], dtype=np.float32))
        out = graph.new_value("out")
        graph.add_node(Node("fused0", "fused_linear_relu", [inp, weight, bias], [out]))
        graph.outputs = [out]

        lowered = TritonLowerer().lower(graph)

        self.assertTrue(lowered.supported)
        self.assertIn("launch fused linear+relu kernel", lowered.execution_plan[0])


@unittest.skipUnless(HAS_TORCH, "torch is required for Triton executor tests")
class TritonExecutorTests(unittest.TestCase):
    def test_executor_reports_unavailable_without_cuda(self) -> None:
        executor = TritonExecutor()
        if torch.cuda.is_available():
            self.skipTest("CUDA is available; unavailability path not applicable.")
        self.assertFalse(executor.available())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import unittest
from typing import Any

import numpy as np

from backend.triton.executor import TritonExecutor
from backend.triton.lowering import TritonLowerer
from backend.triton.strategy import BackendStrategySelector
from ir.graph import Graph
from ir.node import Node


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch


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
        self.assertEqual(lowered.strategy_decisions[0].route, "triton")
        self.assertEqual(lowered.kernel_specs[0].kind, "triton")
        self.assertIn("launch fused linear+relu kernel", lowered.execution_plan[0])

    def test_strategy_selection_rules_for_core_gpu_ops(self) -> None:
        selector = BackendStrategySelector()
        graph = Graph(name="routes")
        x = graph.add_input("x")
        y = graph.add_input("y")
        w = graph.add_constant("w", np.array([[1.0, -1.0]], dtype=np.float32))
        b = graph.add_constant("b", np.array([0.5], dtype=np.float32))
        matmul_out = graph.new_value("matmul_out")
        add_out = graph.new_value("add_out")
        relu_out = graph.new_value("relu_out")
        linear_out = graph.new_value("linear_out")
        fused_out = graph.new_value("fused_out")
        nodes = [
            Node("matmul0", "matmul", [x, y], [matmul_out]),
            Node("add0", "add", [x, y], [add_out]),
            Node("relu0", "relu", [add_out], [relu_out]),
            Node("linear0", "linear", [x, w, b], [linear_out]),
            Node("fused0", "fused_linear_relu", [x, w, b], [fused_out]),
        ]

        routes = {node.name: selector.select(node).route for node in nodes}

        self.assertEqual(routes["matmul0"], "library")
        self.assertEqual(routes["add0"], "generic_gpu")
        self.assertEqual(routes["relu0"], "generic_gpu")
        self.assertEqual(routes["linear0"], "library")
        self.assertEqual(routes["fused0"], "triton")


@unittest.skipUnless(HAS_TORCH, "torch is required for Triton executor tests")
class TritonExecutorTests(unittest.TestCase):
    def test_executor_reports_unavailable_without_cuda(self) -> None:
        executor = TritonExecutor()
        if torch.cuda.is_available():
            self.skipTest("CUDA is available; unavailability path not applicable.")
        self.assertFalse(executor.available())

    def test_executor_runs_fused_graph_with_mock_device_transfer(self) -> None:
        class MockDeviceTritonExecutor(TritonExecutor):
            def available(self) -> bool:
                return True

            def _to_device_tensor(self, value: Any) -> "torch.Tensor":
                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                return tensor

        graph = Graph(name="mock_triton_exec")
        inp = graph.add_input("x")
        weight = graph.add_constant("w", np.array([[1.0, -1.0]], dtype=np.float32))
        bias = graph.add_constant("b", np.array([0.5], dtype=np.float32))
        out = graph.new_value("out")
        graph.add_node(Node("fused0", "fused_linear_relu", [inp, weight, bias], [out]))
        graph.outputs = [out]

        result = MockDeviceTritonExecutor().run(graph, {"x": np.array([[2.0, 1.0]], dtype=np.float32)})

        self.assertTrue(np.allclose(result[0], np.array([[1.5]], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

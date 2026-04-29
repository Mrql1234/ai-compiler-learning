from __future__ import annotations

import unittest

import numpy as np

from backend.cpu.executor import CPUExecutor
from ir.graph import Graph
from ir.node import Node
from passes.fusion import FusionPass


class FusionTests(unittest.TestCase):
    def test_fuses_linear_relu(self) -> None:
        graph = Graph(name="fusion")
        inp = graph.add_input("x")
        weight = graph.add_constant("w", np.array([[1.0, -1.0]], dtype=np.float32))
        bias = graph.add_constant("b", np.array([0.5], dtype=np.float32))
        linear_out = graph.new_value("linear_out")
        relu_out = graph.new_value("relu_out")
        graph.add_node(Node("linear0", "linear", [inp, weight, bias], [linear_out]))
        graph.add_node(Node("relu0", "relu", [linear_out], [relu_out]))
        graph.outputs = [relu_out]

        changed = FusionPass().run(graph)

        self.assertTrue(changed)
        self.assertEqual([node.op_type for node in graph.nodes if node.op_type not in {"constant"}], ["fused_linear_relu"])

    def test_fused_linear_relu_executes(self) -> None:
        graph = Graph(name="fusion_exec")
        inp = graph.add_input("x")
        weight = graph.add_constant("w", np.array([[1.0, -1.0]], dtype=np.float32))
        bias = graph.add_constant("b", np.array([0.5], dtype=np.float32))
        out = graph.new_value("out")
        graph.add_node(Node("fused0", "fused_linear_relu", [inp, weight, bias], [out]))
        graph.outputs = [out]

        result = CPUExecutor().run(graph, {"x": np.array([[2.0, 1.0]], dtype=np.float32)})
        self.assertTrue(np.allclose(result[0], np.array([[1.5]], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

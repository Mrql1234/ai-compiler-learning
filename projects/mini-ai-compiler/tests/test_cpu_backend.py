from __future__ import annotations

import unittest

import numpy as np

from backend.cpu.executor import CPUExecutor
from ir.graph import Graph
from ir.node import Node


class CPUBackendTests(unittest.TestCase):
    def test_add_execution(self) -> None:
        graph = Graph(name="cpu")
        inp = graph.add_input("x")
        const = graph.add_constant("c0", np.array([1.0, 2.0], dtype=np.float32))
        out = graph.new_value("out")
        graph.add_node(Node("add0", "add", [inp, const], [out]))
        graph.outputs = [out]

        result = CPUExecutor().run(graph, {"x": np.array([3.0, 4.0], dtype=np.float32)})

        self.assertTrue(np.allclose(result[0], np.array([4.0, 6.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

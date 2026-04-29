from __future__ import annotations

import unittest

import numpy as np

from ir.graph import Graph
from ir.mlir_printer import print_mlir
from ir.node import Node


class MLIRExportTests(unittest.TestCase):
    def test_prints_module_and_func(self) -> None:
        graph = Graph(name="demo")
        inp = graph.add_input("x")
        const = graph.add_constant("c0", np.array([1.0], dtype=np.float32))
        out = graph.new_value("out")
        graph.add_node(Node("add0", "add", [inp, const], [out]))
        graph.outputs = [out]

        text = print_mlir(graph)

        self.assertIn("module {", text)
        self.assertIn("func.func @demo", text)
        self.assertIn("mini.constant", text)
        self.assertIn("mini.add", text)
        self.assertIn("return", text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from ir.graph import Graph
from ir.node import Node
from passes.mlir_canonicalize import MLIRCanonicalizePass


class MLIRCanonicalizeTests(unittest.TestCase):
    def test_fold_add_const_pattern(self) -> None:
        graph = Graph(name="fold_add")
        lhs = graph.add_constant("lhs", np.array(2))
        rhs = graph.add_constant("rhs", np.array(3))
        out = graph.new_value("out")
        graph.add_node(Node("add0", "add", [lhs, rhs], [out]))
        graph.outputs = [out]

        changed = MLIRCanonicalizePass().run(graph)

        self.assertTrue(changed)
        self.assertTrue(graph.outputs[0].is_constant)
        self.assertEqual(graph.outputs[0].producer.attrs["value"], 5)

    def test_fold_add_zero_pattern(self) -> None:
        graph = Graph(name="fold_zero")
        inp = graph.add_input("x")
        zero = graph.add_constant("zero", np.array(0, dtype=np.int32))
        out = graph.new_value("out")
        graph.add_node(Node("add0", "add", [inp, zero], [out]))
        graph.outputs = [out]

        changed = MLIRCanonicalizePass().run(graph)

        self.assertTrue(changed)
        self.assertIs(graph.outputs[0], inp)

    def test_fold_mul_one_pattern(self) -> None:
        graph = Graph(name="fold_one")
        inp = graph.add_input("x")
        one = graph.add_constant("one", np.array(1, dtype=np.int32))
        out = graph.new_value("out")
        graph.add_node(Node("mul0", "mul", [inp, one], [out]))
        graph.outputs = [out]

        changed = MLIRCanonicalizePass().run(graph)

        self.assertTrue(changed)
        self.assertIs(graph.outputs[0], inp)


if __name__ == "__main__":
    unittest.main()

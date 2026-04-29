from __future__ import annotations

import unittest

import numpy as np

from ir.graph import Graph
from ir.node import Node
from passes.constant_fold import ConstantFoldPass
from passes.dce import DCEPass


class PassTests(unittest.TestCase):
    def test_constant_fold_add(self) -> None:
        graph = Graph(name="fold")
        lhs = graph.add_constant("lhs", np.array(2))
        rhs = graph.add_constant("rhs", np.array(3))
        out = graph.new_value("out")
        graph.add_node(Node("add0", "add", [lhs, rhs], [out]))
        graph.outputs = [out]

        changed = ConstantFoldPass().run(graph)

        self.assertTrue(changed)
        self.assertTrue(graph.outputs[0].is_constant)
        self.assertEqual(graph.outputs[0].producer.attrs["value"], 5)

    def test_dce_removes_dead_node(self) -> None:
        graph = Graph(name="dce")
        lhs = graph.add_constant("lhs", np.array(2))
        rhs = graph.add_constant("rhs", np.array(3))
        dead_out = graph.new_value("dead")
        live_out = graph.new_value("live")
        graph.add_node(Node("dead_add", "add", [lhs, rhs], [dead_out]))
        graph.add_node(Node("live_mul", "mul", [lhs, rhs], [live_out]))
        graph.outputs = [live_out]

        changed = DCEPass().run(graph)

        self.assertTrue(changed)
        self.assertEqual([node.name for node in graph.nodes], ["lhs", "rhs", "live_mul"])


if __name__ == "__main__":
    unittest.main()

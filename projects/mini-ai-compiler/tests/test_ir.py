from __future__ import annotations

import unittest

import numpy as np

from ir.graph import Graph
from ir.node import Node


class IRTests(unittest.TestCase):
    def test_replace_all_uses(self) -> None:
        graph = Graph(name="test")
        const1 = graph.add_constant("c1", np.array(2))
        const2 = graph.add_constant("c2", np.array(3))
        out = graph.new_value("out")
        add = Node("add0", "add", [const1, const1], [out])
        graph.add_node(add)
        graph.outputs = [out]

        graph.replace_all_uses(const1, const2)

        self.assertIs(add.inputs[0], const2)
        self.assertIs(add.inputs[1], const2)


if __name__ == "__main__":
    unittest.main()

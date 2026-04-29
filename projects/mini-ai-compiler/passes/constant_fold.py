from __future__ import annotations

import operator

import numpy as np

from ir.graph import Graph
from ir.node import Node
from passes.base import Pass


class ConstantFoldPass(Pass):
    _evaluators = {
        "add": operator.add,
        "mul": operator.mul,
    }

    def run(self, graph: Graph) -> bool:
        changed = False
        local_change = True
        while local_change:
            local_change = False
            for node in list(graph.nodes):
                if node.op_type not in self._evaluators:
                    continue
                if not all(input_value.is_constant for input_value in node.inputs):
                    continue
                result = self._evaluate(node)
                constant = graph.add_constant(f"{node.name}_folded", result)
                graph.replace_all_uses(node.outputs[0], constant)
                graph.erase_node(node)
                local_change = True
                changed = True
                break
        return changed

    def _evaluate(self, node: Node) -> np.ndarray:
        values = [input_value.producer.attrs["value"] for input_value in node.inputs]
        return self._evaluators[node.op_type](*values)

from __future__ import annotations

from ir.graph import Graph
from ir.node import Node
from passes.base import Pass


class FusionPass(Pass):
    def run(self, graph: Graph) -> bool:
        changed = False

        for node in list(graph.nodes):
            if node.op_type != "relu":
                continue
            if len(node.inputs) != 1:
                continue

            producer = node.inputs[0].producer
            if producer is None or producer.op_type != "linear":
                continue
            if len(node.inputs[0].users) != 1:
                continue

            fused_output = graph.new_value(node.outputs[0].name)
            fused_node = Node(
                name=f"{producer.name}_relu_fused",
                op_type="fused_linear_relu",
                inputs=list(producer.inputs),
                outputs=[fused_output],
                attrs={"fused_from": [producer.op_type, node.op_type]},
            )

            insert_at = graph.nodes.index(producer)
            graph.nodes.insert(insert_at, fused_node)
            graph.replace_all_uses(node.outputs[0], fused_output)
            graph.erase_node(node)
            graph.erase_node(producer)
            changed = True
            break

        return changed

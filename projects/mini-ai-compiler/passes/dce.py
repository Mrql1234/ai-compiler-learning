from __future__ import annotations

from ir.graph import Graph
from passes.base import Pass


class DCEPass(Pass):
    def run(self, graph: Graph) -> bool:
        changed = False
        live_value_ids = {id(value) for value in graph.outputs}
        live_node_ids: set[int] = set()

        worklist = list(graph.outputs)
        while worklist:
            value = worklist.pop()
            producer = value.producer
            if producer is None or id(producer) in live_node_ids:
                continue
            live_node_ids.add(id(producer))
            for input_value in producer.inputs:
                if id(input_value) not in live_value_ids:
                    live_value_ids.add(id(input_value))
                    worklist.append(input_value)

        for node in list(graph.nodes):
            if id(node) not in live_node_ids and node.op_type != "constant":
                graph.erase_node(node)
                changed = True

        for node in list(graph.nodes):
            if node.op_type == "constant" and id(node) not in live_node_ids:
                graph.erase_node(node)
                changed = True

        return changed

from __future__ import annotations

from ir.graph import Graph
from passes.base import Pass


class PassManager:
    def __init__(self, passes: list[Pass]) -> None:
        self.passes = passes

    def run(self, graph: Graph) -> Graph:
        for optimization in self.passes:
            local_change = True
            while local_change:
                local_change = optimization.run(graph)
        return graph

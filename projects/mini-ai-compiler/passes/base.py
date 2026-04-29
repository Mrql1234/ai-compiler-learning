from __future__ import annotations

from ir.graph import Graph


class Pass:
    def run(self, graph: Graph) -> bool:
        raise NotImplementedError

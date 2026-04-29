from __future__ import annotations

from dataclasses import dataclass

from ir.graph import Graph
from ir.node import Node
from ir.value import Value


@dataclass
class RewriteResult:
    changed: bool


class PatternRewriter:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def replace_op_with_constant(self, node: Node, value) -> Value:
        constant = self.graph.add_constant(f"{node.name}_rewritten", value)
        self.graph.replace_all_uses(node.outputs[0], constant)
        self.graph.erase_node(node)
        return constant

    def replace_op(self, node: Node, new_node: Node) -> None:
        insert_at = self.graph.nodes.index(node)
        self.graph.nodes.insert(insert_at, new_node)
        if node.outputs and new_node.outputs:
            self.graph.replace_all_uses(node.outputs[0], new_node.outputs[0])
        self.graph.erase_node(node)

    def erase_op(self, node: Node) -> None:
        self.graph.erase_node(node)


class RewritePattern:
    root_op_type: str | None = None

    def match_and_rewrite(self, node: Node, rewriter: PatternRewriter) -> bool:
        raise NotImplementedError


class GreedyPatternRewriteDriver:
    def __init__(self, patterns: list[RewritePattern]) -> None:
        self.patterns = patterns

    def run(self, graph: Graph) -> bool:
        changed = False
        local_change = True

        while local_change:
            local_change = False
            for node in list(graph.nodes):
                for pattern in self.patterns:
                    if pattern.root_op_type is not None and node.op_type != pattern.root_op_type:
                        continue
                    rewriter = PatternRewriter(graph)
                    if pattern.match_and_rewrite(node, rewriter):
                        changed = True
                        local_change = True
                        break
                if local_change:
                    break

        return changed

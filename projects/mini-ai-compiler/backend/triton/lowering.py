from __future__ import annotations

from dataclasses import dataclass, field

from ir.graph import Graph
from ir.node import Node


@dataclass
class TritonLoweringResult:
    supported: bool
    unsupported_ops: list[str] = field(default_factory=list)
    lowered_nodes: list[str] = field(default_factory=list)
    execution_plan: list[str] = field(default_factory=list)


class TritonLowerer:
    _supported_ops = {
        "matmul",
        "add",
        "relu",
        "linear",
        "fused_linear_relu",
    }

    def lower(self, graph: Graph) -> TritonLoweringResult:
        unsupported_ops: list[str] = []
        lowered_nodes: list[str] = []
        execution_plan: list[str] = []

        for node in graph.nodes:
            if node.op_type == "constant":
                continue
            if node.op_type not in self._supported_ops:
                unsupported_ops.append(node.op_type)
                continue
            lowered_nodes.append(node.name)
            execution_plan.append(self._describe_node(node))

        return TritonLoweringResult(
            supported=len(unsupported_ops) == 0,
            unsupported_ops=unsupported_ops,
            lowered_nodes=lowered_nodes,
            execution_plan=execution_plan,
        )

    def _describe_node(self, node: Node) -> str:
        if node.op_type == "fused_linear_relu":
            return f"{node.name}: launch fused linear+relu kernel"
        if node.op_type == "linear":
            return f"{node.name}: launch linear kernel"
        if node.op_type == "matmul":
            return f"{node.name}: launch matmul kernel"
        if node.op_type == "add":
            return f"{node.name}: launch elementwise add kernel"
        if node.op_type == "relu":
            return f"{node.name}: launch relu kernel"
        return f"{node.name}: unsupported"

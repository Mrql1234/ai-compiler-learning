from __future__ import annotations

from dataclasses import dataclass, field

from backend.triton.kernels import TritonKernelSpec
from backend.triton.strategy import BackendStrategyDecision, BackendStrategySelector
from ir.graph import Graph
from ir.node import Node


@dataclass
class TritonLoweringResult:
    supported: bool
    unsupported_ops: list[str] = field(default_factory=list)
    lowered_nodes: list[str] = field(default_factory=list)
    strategy_decisions: list[BackendStrategyDecision] = field(default_factory=list)
    kernel_specs: list[TritonKernelSpec] = field(default_factory=list)
    execution_plan: list[str] = field(default_factory=list)


class TritonLowerer:
    def __init__(self) -> None:
        self.selector = BackendStrategySelector()

    def lower(self, graph: Graph) -> TritonLoweringResult:
        unsupported_ops: list[str] = []
        lowered_nodes: list[str] = []
        strategy_decisions: list[BackendStrategyDecision] = []
        kernel_specs: list[TritonKernelSpec] = []
        execution_plan: list[str] = []

        for node in graph.nodes:
            if node.op_type == "constant":
                continue
            decision = self.selector.select(node)
            strategy_decisions.append(decision)
            if decision.route == "unsupported":
                unsupported_ops.append(node.op_type)
                continue
            lowered_nodes.append(node.name)
            kernel_specs.append(self._build_kernel_spec(node, decision))
            execution_plan.append(self._describe_node(node, decision))

        return TritonLoweringResult(
            supported=len(unsupported_ops) == 0,
            unsupported_ops=unsupported_ops,
            lowered_nodes=lowered_nodes,
            strategy_decisions=strategy_decisions,
            kernel_specs=kernel_specs,
            execution_plan=execution_plan,
        )

    def _build_kernel_spec(self, node: Node, decision: BackendStrategyDecision) -> TritonKernelSpec:
        return TritonKernelSpec(
            name=node.name,
            kind=decision.route,
            meta={
                "op_type": node.op_type,
                "inputs": len(node.inputs),
                "outputs": len(node.outputs),
                "reason": decision.reason,
            },
        )

    def _describe_node(self, node: Node, decision: BackendStrategyDecision) -> str:
        if node.op_type == "fused_linear_relu":
            action = "launch fused linear+relu kernel"
        elif node.op_type == "linear":
            action = "dispatch library-backed linear kernel"
        elif node.op_type == "matmul":
            action = "dispatch library-backed matmul kernel"
        elif node.op_type == "add":
            action = "lower to generic GPU add kernel"
        elif node.op_type == "relu":
            action = "lower to generic GPU relu kernel"
        else:
            action = "unsupported"
        return f"{node.name}: route={decision.route} -> {action}"

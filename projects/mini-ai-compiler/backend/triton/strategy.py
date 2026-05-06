from __future__ import annotations

from dataclasses import dataclass

from ir.node import Node


@dataclass(frozen=True)
class BackendStrategyDecision:
    node_name: str
    op_type: str
    route: str
    reason: str


class BackendStrategySelector:
    def select(self, node: Node) -> BackendStrategyDecision:
        if node.op_type == "fused_linear_relu":
            return BackendStrategyDecision(
                node_name=node.name,
                op_type=node.op_type,
                route="triton",
                reason="fused epilogue is a good fit for a custom Triton kernel",
            )
        if node.op_type in {"linear", "matmul"}:
            return BackendStrategyDecision(
                node_name=node.name,
                op_type=node.op_type,
                route="library",
                reason="dense GEMM-like ops should start from a library-backed GPU path",
            )
        if node.op_type in {"add", "relu"}:
            return BackendStrategyDecision(
                node_name=node.name,
                op_type=node.op_type,
                route="generic_gpu",
                reason="simple elementwise ops can use the generic GPU path first",
            )
        return BackendStrategyDecision(
            node_name=node.name,
            op_type=node.op_type,
            route="unsupported",
            reason="no GPU strategy rule is registered for this op yet",
        )

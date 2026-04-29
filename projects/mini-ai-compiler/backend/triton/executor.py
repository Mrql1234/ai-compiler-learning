from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from backend.triton.kernels import (
    add_reference,
    linear_reference,
    linear_relu_reference,
    relu_reference,
    triton_available,
)
from backend.triton.lowering import TritonLowerer
from ir.graph import Graph


class TritonExecutor:
    def __init__(self) -> None:
        self.lowerer = TritonLowerer()

    def available(self) -> bool:
        return triton_available()

    def run(self, graph: Graph, inputs: dict[str, Any]) -> list[Any]:
        if torch is None:
            raise RuntimeError("torch is required for Triton backend execution.")

        lowered = self.lowerer.lower(graph)
        if not lowered.supported:
            raise RuntimeError(
                "Unsupported ops for Triton backend: " + ", ".join(sorted(set(lowered.unsupported_ops)))
            )
        if not self.available():
            raise RuntimeError("Triton backend requires a CUDA-capable PyTorch runtime.")

        env: dict[str, Any] = {}
        for value in graph.inputs:
            env[value.name] = self._to_device_tensor(inputs[value.name])

        for node in graph.nodes:
            if node.op_type == "constant":
                env[node.outputs[0].name] = self._to_device_tensor(node.attrs["value"])
                continue

            args = [env[input_value.name] for input_value in node.inputs]
            if node.op_type == "matmul":
                result = args[0] @ args[1]
            elif node.op_type == "add":
                result = add_reference(args)
            elif node.op_type == "relu":
                result = relu_reference(args)
            elif node.op_type == "linear":
                result = linear_reference(args)
            elif node.op_type == "fused_linear_relu":
                result = linear_relu_reference(args)
            else:  # pragma: no cover
                raise RuntimeError(f"Unsupported Triton op: {node.op_type}")
            env[node.outputs[0].name] = result

        return [env[value.name].detach().cpu().numpy() for value in graph.outputs]

    def _to_device_tensor(self, value: Any) -> "torch.Tensor":
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.cuda()

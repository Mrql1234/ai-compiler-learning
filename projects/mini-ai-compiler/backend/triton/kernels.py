from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class TritonKernelSpec:
    name: str
    kind: str
    meta: dict[str, Any]


def triton_available() -> bool:
    if torch is None:
        return False
    return bool(torch.cuda.is_available())


def linear_relu_reference(inputs: list["torch.Tensor"]) -> "torch.Tensor":
    x, weight, bias = inputs
    result = x @ weight.t()
    result = result + bias
    return torch.relu(result)


def linear_reference(inputs: list["torch.Tensor"]) -> "torch.Tensor":
    if len(inputs) == 2:
        x, weight = inputs
        return x @ weight.t()
    x, weight, bias = inputs
    return x @ weight.t() + bias


def add_reference(inputs: list["torch.Tensor"]) -> "torch.Tensor":
    return inputs[0] + inputs[1]


def relu_reference(inputs: list["torch.Tensor"]) -> "torch.Tensor":
    return torch.relu(inputs[0])

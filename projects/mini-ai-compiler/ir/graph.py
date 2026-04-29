from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .node import Node
from .types import TensorType
from .value import Value


@dataclass
class Graph:
    name: str
    inputs: list[Value] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    _value_index: int = 0

    def add_input(self, name: str, value_type: TensorType | None = None) -> Value:
        value = Value(name=name, type=value_type)
        self.inputs.append(value)
        return value

    def new_value(self, name: str | None = None, value_type: TensorType | None = None) -> Value:
        if name is None:
            name = f"v{self._value_index}"
            self._value_index += 1
        return Value(name=name, type=value_type)

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_constant(self, name: str, data: Any) -> Value:
        value = self.new_value(name=name, value_type=self._infer_tensor_type(data))
        node = Node(name=name, op_type="constant", inputs=[], outputs=[value], attrs={"value": data})
        value.data = data
        self.add_node(node)
        return value

    def replace_all_uses(self, old: Value, new: Value) -> None:
        for user in list(old.users):
            user.replace_input(old, new)
        if old in self.outputs:
            self.outputs = [new if value is old else value for value in self.outputs]

    def erase_node(self, node: Node) -> None:
        if node in self.nodes:
            self.nodes.remove(node)
        for input_value in node.inputs:
            if node in input_value.users:
                input_value.users.remove(node)
        for output in node.outputs:
            output.producer = None
            output.users.clear()

    def _infer_tensor_type(self, data: Any) -> TensorType:
        array = np.asarray(data)
        return TensorType(shape=tuple(int(dim) for dim in array.shape), dtype=str(array.dtype))

    def __str__(self) -> str:
        lines = [f"graph @{self.name}("]
        for value in self.inputs:
            lines.append(f"  input {value}")
        for node in self.nodes:
            lines.append(f"  {node}")
        output_text = ", ".join(f"%{value.name}" for value in self.outputs)
        lines.append(f"  return {output_text}")
        lines.append(")")
        return "\n".join(lines)

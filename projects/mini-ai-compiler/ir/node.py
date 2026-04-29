from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .value import Value


@dataclass
class Node:
    name: str
    op_type: str
    inputs: list[Value] = field(default_factory=list)
    outputs: list[Value] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in self.inputs:
            if self not in value.users:
                value.users.append(self)
        for value in self.outputs:
            value.producer = self

    def replace_input(self, old: Value, new: Value) -> None:
        for index, value in enumerate(self.inputs):
            if value is old:
                self.inputs[index] = new
        if self in old.users:
            old.users.remove(self)
        if self not in new.users:
            new.users.append(self)

    def __str__(self) -> str:
        outputs = ", ".join(f"%{value.name}" for value in self.outputs)
        inputs = ", ".join(f"%{value.name}" for value in self.inputs)
        attrs = ""
        if self.attrs:
            attrs = " {" + ", ".join(f"{key}={value}" for key, value in self.attrs.items()) + "}"
        return f"{outputs} = {self.op_type}({inputs}){attrs}"

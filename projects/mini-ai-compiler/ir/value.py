from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import TensorType


@dataclass
class Value:
    name: str
    type: TensorType | None = None
    producer: "Node | None" = None
    users: list["Node"] = field(default_factory=list)
    data: Any = None

    @property
    def is_constant(self) -> bool:
        return self.producer is not None and self.producer.op_type == "constant"

    def __str__(self) -> str:
        suffix = f": {self.type}" if self.type is not None else ""
        return f"%{self.name}{suffix}"

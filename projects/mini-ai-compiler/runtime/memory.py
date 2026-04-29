from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryPlan:
    buffers: list[str] = field(default_factory=list)

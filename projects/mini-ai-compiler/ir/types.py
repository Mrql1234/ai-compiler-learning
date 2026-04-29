from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TensorType:
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    def __str__(self) -> str:
        if self.shape is None and self.dtype is None:
            return "tensor<?>"
        shape_text = "x".join(str(dim) for dim in self.shape) if self.shape is not None else "?"
        dtype_text = self.dtype or "?"
        return f"tensor<{shape_text}x{dtype_text}>"

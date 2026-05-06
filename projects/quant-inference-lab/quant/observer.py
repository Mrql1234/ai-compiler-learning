from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantizationParams:
    scale: np.ndarray
    zero_point: np.ndarray
    axis: int | None
    symmetric: bool


class MinMaxObserver:
    def __init__(self, symmetric: bool = True, per_channel_axis: int | None = None) -> None:
        self.symmetric = symmetric
        self.per_channel_axis = per_channel_axis
        self.min_val: np.ndarray | None = None
        self.max_val: np.ndarray | None = None

    def observe(self, tensor: np.ndarray) -> None:
        array = np.asarray(tensor, dtype=np.float32)
        if self.per_channel_axis is None:
            current_min = np.array(array.min(), dtype=np.float32)
            current_max = np.array(array.max(), dtype=np.float32)
        else:
            reduce_axes = tuple(
                index for index in range(array.ndim) if index != self.per_channel_axis
            )
            current_min = array.min(axis=reduce_axes)
            current_max = array.max(axis=reduce_axes)

        if self.min_val is None:
            self.min_val = current_min
            self.max_val = current_max
            return

        self.min_val = np.minimum(self.min_val, current_min)
        self.max_val = np.maximum(self.max_val, current_max)

    def calculate_qparams(self) -> QuantizationParams:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has not seen any tensors.")

        min_val = self.min_val.astype(np.float32)
        max_val = self.max_val.astype(np.float32)

        if self.symmetric:
            abs_max = np.maximum(np.abs(min_val), np.abs(max_val))
            scale = np.maximum(abs_max / 127.0, 1e-8).astype(np.float32)
            zero_point = np.zeros_like(scale, dtype=np.int32)
        else:
            scale = np.maximum((max_val - min_val) / 255.0, 1e-8).astype(np.float32)
            zero_point = np.round(-min_val / scale).astype(np.int32)
            zero_point = np.clip(zero_point, 0, 255)

        return QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            axis=self.per_channel_axis,
            symmetric=self.symmetric,
        )

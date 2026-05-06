from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observer import MinMaxObserver, QuantizationParams


@dataclass
class QuantizedTensor:
    data: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray
    axis: int | None
    symmetric: bool

    def dequantize(self) -> np.ndarray:
        if self.axis is None:
            return (self.data.astype(np.float32) - self.zero_point) * self.scale

        reshape = [1] * self.data.ndim
        reshape[self.axis] = self.scale.shape[0]
        scale = self.scale.reshape(reshape)
        zero_point = self.zero_point.reshape(reshape)
        return (self.data.astype(np.float32) - zero_point) * scale


def _broadcast_params(params: QuantizationParams, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if params.axis is None:
        return params.scale, params.zero_point

    reshape = [1] * array.ndim
    reshape[params.axis] = params.scale.shape[0]
    return params.scale.reshape(reshape), params.zero_point.reshape(reshape)


def quantize_tensor(
    tensor: np.ndarray,
    *,
    symmetric: bool = True,
    per_channel_axis: int | None = None,
) -> QuantizedTensor:
    array = np.asarray(tensor, dtype=np.float32)
    observer = MinMaxObserver(symmetric=symmetric, per_channel_axis=per_channel_axis)
    observer.observe(array)
    params = observer.calculate_qparams()
    scale, zero_point = _broadcast_params(params, array)

    if symmetric:
        quantized = np.round(array / scale)
        quantized = np.clip(quantized, -127, 127).astype(np.int8)
    else:
        quantized = np.round(array / scale + zero_point)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

    return QuantizedTensor(
        data=quantized,
        scale=params.scale,
        zero_point=params.zero_point,
        axis=params.axis,
        symmetric=params.symmetric,
    )

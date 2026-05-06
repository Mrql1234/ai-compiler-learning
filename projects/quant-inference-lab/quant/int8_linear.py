from __future__ import annotations

import numpy as np

from .quantizer import QuantizedTensor, quantize_tensor


def linear_fp32(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    output = np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
    if bias is not None:
        output = output + np.asarray(bias, dtype=np.float32)
    return output


def linear_int8_weight_only(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    per_channel_axis: int = 0,
) -> tuple[np.ndarray, QuantizedTensor]:
    qweight = quantize_tensor(weight, symmetric=True, per_channel_axis=per_channel_axis)
    output = linear_fp32(x, qweight.dequantize(), bias)
    return output, qweight


def linear_int8_dynamic_input(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    input_symmetric: bool = False,
    weight_per_channel_axis: int = 0,
) -> tuple[np.ndarray, QuantizedTensor, QuantizedTensor]:
    qinput = quantize_tensor(x, symmetric=input_symmetric, per_channel_axis=None)
    qweight = quantize_tensor(weight, symmetric=True, per_channel_axis=weight_per_channel_axis)
    output = linear_fp32(qinput.dequantize(), qweight.dequantize(), bias)
    return output, qinput, qweight

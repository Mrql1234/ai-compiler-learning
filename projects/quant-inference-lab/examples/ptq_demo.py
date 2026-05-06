from __future__ import annotations

import numpy as np

from quant import linear_fp32, linear_int8_dynamic_input, linear_int8_weight_only


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 16)).astype(np.float32)
    weight = rng.normal(size=(8, 16)).astype(np.float32)
    bias = rng.normal(size=(8,)).astype(np.float32)

    fp32 = linear_fp32(x, weight, bias)
    weight_only, qweight = linear_int8_weight_only(x, weight, bias)
    dynamic, qinput, _ = linear_int8_dynamic_input(x, weight, bias)

    print("=== PTQ Demo ===")
    print(f"weight-only scale shape: {qweight.scale.shape}")
    print(f"dynamic-input scale shape: {qinput.scale.shape}")
    print(f"weight-only max abs error: {np.max(np.abs(fp32 - weight_only)):.6f}")
    print(f"dynamic-input max abs error: {np.max(np.abs(fp32 - dynamic)):.6f}")
    print(f"fp32[0, :4]        = {np.array2string(fp32[0, :4], precision=4)}")
    print(f"weight-only[0, :4] = {np.array2string(weight_only[0, :4], precision=4)}")
    print(f"dynamic[0, :4]     = {np.array2string(dynamic[0, :4], precision=4)}")


if __name__ == "__main__":
    main()

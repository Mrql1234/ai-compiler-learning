from __future__ import annotations

import unittest

import numpy as np

from quant import linear_fp32, linear_int8_weight_only, quantize_tensor


class QuantizerTests(unittest.TestCase):
    def test_symmetric_per_tensor_roundtrip_is_reasonable(self) -> None:
        array = np.array([-1.5, -0.1, 0.2, 1.7], dtype=np.float32)
        quantized = quantize_tensor(array, symmetric=True)
        restored = quantized.dequantize()
        self.assertLess(np.max(np.abs(array - restored)), 0.05)

    def test_per_channel_scale_shape_matches_axis(self) -> None:
        weight = np.ones((8, 16), dtype=np.float32)
        quantized = quantize_tensor(weight, symmetric=True, per_channel_axis=0)
        self.assertEqual(quantized.scale.shape, (8,))

    def test_weight_only_linear_tracks_fp32(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 16)).astype(np.float32)
        w = rng.normal(size=(8, 16)).astype(np.float32)
        b = rng.normal(size=(8,)).astype(np.float32)
        ref = linear_fp32(x, w, b)
        out, _ = linear_int8_weight_only(x, w, b)
        self.assertLess(np.max(np.abs(ref - out)), 0.2)


if __name__ == "__main__":
    unittest.main()

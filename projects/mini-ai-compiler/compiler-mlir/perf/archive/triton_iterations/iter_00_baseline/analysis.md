# Iteration Analysis

这一轮用于建立 Triton baseline。

- case: `triton_linear_relu_f32_m1024_n1024_k1024`
- config: `bm64_bn64_bk32_gm8_w4_s2`
- correctness: `yes`
- kernel_ms.median: `0.413696 ms`
- invoke_ms.median: `0.429729 ms`

用途：作为后续 sweep 和 profile 的性能对照。


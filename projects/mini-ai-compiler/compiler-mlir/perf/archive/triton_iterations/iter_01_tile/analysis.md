# Iteration Analysis

这一轮围绕 tile / pipeline 参数做 sweep。

- candidate_count: `144`
- failed_candidate_count: `24`
- best_config_tag: `bm128_bn128_bk32_gm4_w8_s3`
- best_kernel_ms.median: `0.254976 ms`
- speedup_vs_baseline: `1.622x`

结论：`128x128x32` 大 tile 明显优于原始 `64x64x32`，`GROUP_M=4` 与 `num_warps=8` / `num_stages=3` 组合目前最好。


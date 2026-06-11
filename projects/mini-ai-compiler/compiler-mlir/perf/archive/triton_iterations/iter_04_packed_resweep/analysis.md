# Iteration Analysis

这一轮围绕 tile / pipeline 参数做 sweep。

- candidate_count: `144`
- failed_candidate_count: `24`
- best_config_tag: `bm128_bn128_bk32_gm4_w8_s4`
- best_kernel_ms.median: `0.159648 ms`
- speedup_vs_baseline: `2.591x`

结论：`128x128x32` 大 tile 仍然是当前主优区域，本轮名义最优配置是 `bm128_bn128_bk32_gm4_w8_s4`。
如果前几名差距非常小，仍需要补单点复测，再决定是否更新默认配置。


# Iteration Analysis

这一轮做 `nsys + ncu` profile，用于定位下一轮优化方向。

- profiled_bench_kernel_ms.median: `1680.202087 ms`
- profiled_bench_invoke_ms.median: `1680.444142 ms`
- reference_kernel_ms.median: `0.254976 ms`
- reference_invoke_ms.median: `0.270839 ms`

说明：profiled benchmark 数值会被 profiler 显著放大，性能结论应以 reference benchmark 为准。
- registers_per_thread: `236.0`
- dynamic_shared_memory_kib: `65.54`
- achieved_occupancy_pct: `16.66`
- no_eligible_pct: `61.96`
- l2_hit_rate_pct: `88.25`
- shared_bank_conflicts: `14680064`
- shared_bank_conflict_wavefront_pct: `69.89`
- short_scoreboard_cycles: `2.4`

结论：当前最突出的问题是 shared-memory bank conflict 和低 eligible warps，而不是 DRAM 带宽本身。


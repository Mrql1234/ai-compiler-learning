# Iteration Analysis

这一轮做 `nsys + ncu` profile，用于定位下一轮优化方向。

- profiled_bench_kernel_ms.median: `0.192928 ms`
- profiled_bench_invoke_ms.median: `0.266102 ms`
- reference_kernel_ms.median: `0.156672 ms`
- reference_invoke_ms.median: `0.177014 ms`

说明：profiled benchmark 数值会被 profiler 显著放大，性能结论应以 reference benchmark 为准。
- duration_us: `234.75`
- memory_throughput_gbps: `58.06`
- compute_sm_throughput_pct: `63.95`
- registers_per_thread: `186.0`
- dynamic_shared_memory_kib: `65.54`
- achieved_occupancy_pct: `16.66`
- no_eligible_pct: `27.01`
- eligible_warps_per_scheduler: `1.29`
- l2_hit_rate_pct: `88.23`
- shared_bank_conflicts: `0`
- shared_bank_conflict_wavefront_pct: `None`
- short_scoreboard_cycles: `None`

结论：当前目标 kernel 已不再出现显式 shared-memory bank-conflict 提示，下一轮更应关注低 occupancy 下的发射效率和资源压力平衡。


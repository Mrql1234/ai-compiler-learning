# NCU Metrics Guide

## Contents

1. Core metric set
2. Bottleneck decision rules
3. Metric-by-metric interpretation
4. Common misreads
5. A10 / Triton notes

## 1. Core Metric Set

Start with these metrics before reading anything else:

- kernel duration or `kernel_ms`
- achieved occupancy
- SM throughput
- DRAM throughput
- L2 hit rate or throughput
- registers per thread
- shared memory per CTA
- top warp stall reasons

If you do not have most of this set, avoid overconfident conclusions.

## 2. Bottleneck Decision Rules

Use these heuristics:

- `memory-bound`
  - DRAM throughput is high relative to the rest
  - SM throughput is clearly lower
  - stalls often point to memory dependencies or long scoreboard

- `compute-bound`
  - SM throughput is high
  - memory metrics are not the dominant limiter
  - stalls may point to math pipe or issue slot pressure

- `latency-hiding-limited`
  - occupancy is low or modest
  - memory-latency stalls are visible
  - resource pressure may be limiting active warps

- `resource-limited`
  - registers per thread or shared memory per CTA is large
  - achieved occupancy drops accordingly

- `under-filled kernel`
  - tiny workload, tiny grid, or too few resident blocks
  - throughput looks low simply because the GPU is not busy enough

## 3. Metric-By-Metric Interpretation

### Achieved Occupancy

What it means:

- how many warps are active relative to hardware capacity

How to use it:

- low occupancy can make latency hiding worse
- low occupancy does not automatically mean the kernel is badly optimized

Common Triton implications:

- tiles too large
- `num_warps` too high
- `num_stages` too high
- register or shared-memory footprint too large

### SM Throughput

What it means:

- how busy the compute pipelines are

How to use it:

- low SM throughput with high memory pressure often means the kernel is not compute-limited
- high SM throughput with acceptable memory behavior suggests compute pressure is more important

### DRAM Throughput

What it means:

- how much external memory bandwidth the kernel is using

How to use it:

- high DRAM throughput with weak SM utilization often points to memory-bound behavior
- low DRAM throughput does not guarantee the kernel is compute-bound; it may just be under-filled

### L2 Hit Rate / Throughput

What it means:

- how effectively the kernel reuses data from cache

How to use it:

- poor L2 behavior may suggest bad tile ordering or weak locality
- grouped ordering changes can matter here

### Registers Per Thread

What it means:

- per-thread register footprint

How to use it:

- high register count can lower occupancy
- large tiles and aggressive pipeline depth often increase this

### Shared Memory Per CTA

What it means:

- on-chip shared memory used by each block / CTA

How to use it:

- high shared-memory use can limit concurrent CTAs
- this can indirectly reduce occupancy and latency hiding

### Warp Stall Reasons

These usually give the best hints for what to tune next.

Common examples:

- `long scoreboard`
  - often memory latency or dependency waiting
- `not selected`
  - often enough eligible warps exist; not always a problem
- `barrier`
  - synchronization overhead matters
- `math pipe throttle`
  - arithmetic pipeline pressure may be important

## 4. Common Misreads

Do not make these mistakes:

- `low occupancy => kernel is bad`
- `high DRAM throughput => everything is fine`
- `high SM throughput => no further optimization needed`
- `single stall reason => single root cause`

Always cross-check:

- occupancy + resource usage
- throughput + stalls
- benchmark time + workload size

## 5. A10 / Triton Notes

For Triton kernels on A10-like workflows, the most common tuning loop is:

1. adjust tile sizes
2. re-check register and shared-memory pressure
3. tune `num_warps`
4. tune `num_stages`
5. test program ordering such as `GROUP_M`

When the kernel is small, weak utilization may come from problem size rather than parameter quality.

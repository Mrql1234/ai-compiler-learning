---
name: ncu-analysis
description: Analyze NVIDIA Nsight Compute (`ncu`) results for CUDA, Triton, and MLIR-generated GPU kernels. Use when Codex needs to interpret `ncu` text exports, `.ncu-rep` derived reports, occupancy/throughput/stall metrics, or explain why a kernel is slow and what to tune next. Especially useful for Triton tuning work involving `BLOCK_M/N/K`, `num_warps`, `num_stages`, `GROUP_M`, A10/sm_86 experiments, and profile-driven optimization loops.
---

# NCU Analysis

Use this skill to turn raw Nsight Compute output into an optimization diagnosis.

## Quick Start

If the user provides large `ncu` text exports, first reduce them:

```bash
python3 perf/skills/ncu-analysis/scripts/extract_ncu_key_lines.py \
  path/to/ncu_details.txt
```

Then analyze the reduced output with this flow:

1. Identify the kernel and workload context.
2. Classify the bottleneck from a small set of core metrics.
3. Map the symptoms to likely Triton parameter or kernel-writing changes.
4. Recommend the next 1 to 3 experiments, not a giant list.

## Workflow

### 1. Gather Context

Prefer these artifacts when available:

- `ncu --page details` text export
- `ncu --page session` text export
- benchmark JSON with `kernel_ms`
- kernel source or Triton config
- shape, dtype, GPU model

For Triton tuning, always try to recover:

- `BLOCK_M`
- `BLOCK_N`
- `BLOCK_K`
- `num_warps`
- `num_stages`
- `GROUP_M`

If the user already gave the kernel config, do not ask for it again.

### 2. Reduce The Report

For long reports, use `scripts/extract_ncu_key_lines.py` first.

If you need to grep manually, start with these patterns:

- `Occupancy`
- `SM`
- `DRAM`
- `L2`
- `Registers`
- `Shared Memory`
- `Achieved`
- `Duration`
- `Warp Stall`
- `Launch Statistics`

### 3. Read The Metrics In Order

Read metrics in this order:

1. `kernel_ms` or kernel duration
2. achieved occupancy
3. SM throughput
4. DRAM throughput
5. L2 hit rate / throughput
6. registers per thread
7. shared memory per CTA
8. top warp stall reasons

Use `references/metrics-guide.md` for interpretation.

### 4. Classify The Bottleneck

Use one of these labels early:

- `memory-bound`
- `compute-bound`
- `latency-hiding-limited`
- `resource-limited`
- `under-filled kernel`
- `mixed / inconclusive`

Do not overfit to occupancy alone. Low occupancy is a clue, not the conclusion.

### 5. Map To Triton Changes

Use `references/triton-tuning-map.md` to connect symptoms to likely changes.

Common examples:

- High DRAM pressure + low SM utilization
  - try larger reuse tiles or better program ordering
- High register pressure + low occupancy
  - reduce tile size or `num_warps`
- Memory-latency stalls such as long scoreboard
  - try higher `num_stages` if resource pressure allows
- Cache-locality problems
  - test `GROUP_M` or program mapping changes

### 6. Write The Output

Default output structure:

1. One-sentence bottleneck summary
2. Evidence: 3 to 6 metrics with short interpretation
3. Likely cause in kernel terms
4. Next experiments: 1 to 3 concrete config changes

Prefer phrases like:

- `This kernel currently looks memory-bound.`
- `The main limiter appears to be register pressure rather than raw DRAM bandwidth.`
- `The next experiment should isolate whether num_stages is helping latency hiding or only hurting occupancy.`

## Response Rules

- Always tie conclusions to observed metrics.
- Do not recommend changing many parameters at once unless the user explicitly asks for a broad sweep.
- For Triton kernels, suggest parameter changes in small controlled steps.
- Distinguish `what the metrics say` from `what to try next`.
- If data is missing, say what is missing and what can still be inferred.

## Resources

- `references/metrics-guide.md`
  - metric meaning, bottleneck heuristics, common misreads
- `references/triton-tuning-map.md`
  - symptom-to-parameter mapping for Triton/A10-style tuning
- `scripts/extract_ncu_key_lines.py`
  - extract high-signal lines from large `ncu` text exports

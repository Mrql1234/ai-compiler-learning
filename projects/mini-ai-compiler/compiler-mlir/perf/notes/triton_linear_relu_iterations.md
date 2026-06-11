# Triton `fused_linear_relu` 迭代记录

本文档用于记录 `compiler-mlir` 在 A10 上围绕 Triton `fused_linear_relu` kernel 的多轮优化过程。

当前状态说明：

- 本地环境没有 GPU
- 本文件的结构、命令和记录模板已经准备好
- 真实 benchmark / Nsight 数据需要到云端 A10 环境补齐

配套文件：

- [`../cases/triton_linear_relu_f32_m128_n128_k128.json`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/cases/triton_linear_relu_f32_m128_n128_k128.json)
- [`../cases/triton_linear_relu_f32_m1024_n1024_k1024.json`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json)
- [`../configs/triton_linear_relu_a10.json`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/configs/triton_linear_relu_a10.json)
- [`../CLOUD_TRITON_A10_WORKFLOW.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/CLOUD_TRITON_A10_WORKFLOW.md)

## 记录约定

每一轮都至少记录：

- 改了什么
- 跑了哪个 case
- 用了哪组 config
- `kernel_ms` 怎么变
- Nsight 指标怎么变
- 为什么会这样
- 下一轮准备做什么

## Iteration 0：建立 baseline

目标：

- 跑通第一个 Triton `fused_linear_relu` kernel
- 固定第一版默认 config
- 记录 baseline `kernel_ms`

云端执行命令：

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source default \
  --warmup 10 \
  --repeat 50 \
  --json-output perf/runs/triton_iterations/iter_00_baseline.json
```

结果记录：

- config：
- `kernel_ms.median`：
- `invoke_ms.median`：
- `effective_gflops`：
- `effective_gbps`：
- correctness：

Nsight 观察：

- occupancy：
- SM throughput：
- DRAM throughput：
- 主要 stall：

结论：

- 待云端补充

## Iteration 1：扫 tile

目标：

- 先只关注 `BLOCK_M/BLOCK_N/BLOCK_K`
- 找到更合理的 tile 区间

云端执行命令：

```bash
python3 ./scripts/triton_perf_sweep.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --warmup 10 \
  --repeat 50 \
  --out perf/runs/triton_iterations/iter_01_tile
```

结果记录：

- best config：
- `kernel_ms.median`：
- 对 baseline 改善：
- occupancy 变化：
- registers/shared memory 变化：

结论：

- 待云端补充

## Iteration 2：调 `num_warps` / `num_stages`

目标：

- 在较优 tile 上继续调并行度和流水深度

云端执行命令：

```bash
python3 ./scripts/triton_profile_iter.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source profile_target \
  --tag iter_02_pipeline \
  --out perf/profiles/triton_iterations/iter_02_pipeline
```

结果记录：

- 最优 `num_warps`：
- 最优 `num_stages`：
- `kernel_ms.median`：
- long scoreboard / memory stall 变化：
- occupancy 变化：

结论：

- 待云端补充

## Iteration 3：调 program mapping / cache 行为

目标：

- 聚焦 `GROUP_M`
- 观察 L2 / DRAM 行为变化

建议动作：

- 在相同 tile 下比较 `GROUP_M=4` 和 `GROUP_M=8`
- 观察 `L2 hit rate` 和 `DRAM throughput`

结果记录：

- config：
- `kernel_ms.median`：
- L2 指标变化：
- DRAM 指标变化：
- 主要结论：

## Iteration 4：调 epilogue 融合与写回路径

目标：

- 确认 bias + relu 融合写法是否真的减少了 memory traffic

建议动作：

- 对比当前 fused kernel 与拆分式写回方案
- 聚焦 store 路径和 DRAM 压力

结果记录：

- `kernel_ms.median`：
- DRAM throughput 变化：
- store 相关 stall：
- 主要结论：

## 阶段总结

最终需要补齐：

- Triton 相对 `generated_nvvm` 的提升
- Triton 相对 `cublas` 的距离
- 这组 shape-specialized config 是否值得接回 backend selection

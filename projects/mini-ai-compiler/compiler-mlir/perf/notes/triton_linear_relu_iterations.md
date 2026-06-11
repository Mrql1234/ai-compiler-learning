# Triton `fused_linear_relu` 迭代记录

本文档用于记录 `compiler-mlir` 在 A10 上围绕 Triton `fused_linear_relu` kernel 的多轮优化过程。

当前状态说明：

- 已在云端 A10 环境完成第一轮真实 benchmark / sweep / Nsight 采集
- 本文件不再只是模板，下面的 baseline / sweep / profile 数据都来自真实运行
- 当前重点已经从“搭工作流”进入“根据 profile 结果继续做下一轮 kernel 优化”

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

- config：`BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2`
- `kernel_ms.median`：`0.413696 ms`
- `invoke_ms.median`：`0.433775 ms`
- `effective_gflops`：`5197.730 GFLOP/s`
- `effective_gbps`：`30.424 GB/s`
- correctness：通过

Nsight 观察：

- occupancy：这一轮没有单独做 Nsight profile，先作为纯 benchmark 基线
- SM throughput：待下一轮 profile 补齐
- DRAM throughput：待下一轮 profile 补齐
- 主要 stall：待下一轮 profile 补齐

结论：

- baseline 已经固定，可作为后续 sweep 和 profile 的对照

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

- best config：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=4, num_warps=8, num_stages=3`
- `kernel_ms.median`：`0.254976 ms`
- 对 baseline 改善：`1.62x`（`0.413696 -> 0.254976 ms`）
- occupancy 变化：从 profile 看最优配置理论 occupancy 很低，但 tile 变大后总吞吐明显更高
- registers/shared memory 变化：最优配置在 profile 中已经表现出较高寄存器和 shared memory 压力

结论：

- 大 tile `128x128x32` 明显优于原始 `64x64x32`
- `GROUP_M=4` 比 `GROUP_M=8` 略优，但前 4 名已经非常接近
- 这轮 sweep 一共有 `144` 个候选，其中 `24` 个失败，失败模式主要集中在 `BLOCK_K=64` 且 `num_stages=3/4` 的高资源组合

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

- 最优 `num_warps`：`8`
- 最优 `num_stages`：`3`
- `kernel_ms.median`：standalone benchmark 为 `0.254976 ms`
- long scoreboard / memory stall 变化：这轮更显著的是 `short scoreboard` / shared-memory 相关停顿，平均每条指令间约 `2.4 cycles` 卡在 MIO 依赖
- occupancy 变化：`ncu` 里 Triton `kernel` 的理论 occupancy 只有 `16.67%`，实测 `16.65%`

结论：

- 尽管 occupancy 很低，这组 `8 warps + 3 stages` 仍然是当前 sweep 的最快组合
- 说明当前瓶颈不是简单“把 occupancy 做高”，而是 tile 复用收益和 shared-memory 访存代价之间的平衡

## Iteration 3：调 program mapping / cache 行为

目标：

- 聚焦 `GROUP_M`
- 观察 L2 / DRAM 行为变化

建议动作：

- 在相同 tile 下比较 `GROUP_M=4` 和 `GROUP_M=8`
- 观察 `L2 hit rate` 和 `DRAM throughput`

结果记录：

- config：本轮先观察 `GROUP_M=4` 的最优配置
- `kernel_ms.median`：`0.254976 ms`
- L2 指标变化：`L2 Hit Rate = 88.29%`
- DRAM 指标变化：`Memory Throughput = 31.64 GB/s`，`DRAM Throughput = 5.28%`
- 主要结论：
  - L2 命中率已经不差
  - DRAM 并不是主瓶颈，主要问题更像 shared-memory 访问形态和低 eligible warps

## Iteration 4：调 epilogue 融合与写回路径

目标：

- 确认 bias + relu 融合写法是否真的减少了 memory traffic

建议动作：

- 对比当前 fused kernel 与拆分式写回方案
- 聚焦 store 路径和 DRAM 压力

结果记录：

- `kernel_ms.median`：当前最优 standalone 结果仍为 `0.254976 ms`
- DRAM throughput 变化：从现有 profile 看 DRAM 利用率不高，不像主要瓶颈
- store 相关 stall：当前更突出的不是 store，而是 shared-load bank conflict 和 MIO scoreboard stall
- 主要结论：
- `bias + relu` 融合本身没有暴露出明显额外写回瓶颈
- 下一步优先级应放在 shared-memory 访问布局，而不是继续只调 epilogue

## Iteration 5：权重预打包为 `KxN` 连续布局

目标：

- 直接针对 `ncu` 暴露出的 shared-memory 访问形态问题动手
- 不再只调参数，而是先修正 kernel 读取权重时的布局匹配关系

源码入口：

- [`../../scripts/triton_linear_relu_bench.py`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/scripts/triton_linear_relu_bench.py)

具体改动：

- 新增 `pack_weight_for_kernel(weight_tensor)`，把 `Linear.weight` 从原始 `N x K` 预打包成 `K x N contiguous`
- kernel 仍保留 raw pointer matmul 写法，但 launch 时改为传 packed weight
- 修正权重 stride 传参顺序：
  - `stride_wn = packed_weight_tensor.stride(1)`
  - `stride_wk = packed_weight_tensor.stride(0)`
- 保留 `tl.dot(..., input_precision="ieee")`，确保 correctness 不回退

云端执行命令：

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source profile_target \
  --warmup 10 \
  --repeat 50 \
  --BLOCK_M 128 \
  --BLOCK_N 128 \
  --BLOCK_K 32 \
  --GROUP_M 4 \
  --num-warps 8 \
  --num-stages 3 \
  --json-output perf/runs/triton_iterations/iter_03_packed_bench.json
```

`ncu` workaround 命令：

```bash
./scripts/perf_profile_ncu.sh \
  perf/profiles/triton_iterations/iter_03_packed_layout/iter_03_packed_layout_ncu_full \
  /usr/bin/python3 ./scripts/triton_linear_relu_bench.py \
    --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
    --config perf/configs/triton_linear_relu_a10.json \
    --config-source profile_target \
    --warmup 1 \
    --repeat 2 \
    --device-index 0 \
    --json-output perf/profiles/triton_iterations/iter_03_packed_layout/iter_03_packed_layout_ncu_full_bench.json \
    --emit-nvtx \
    --BLOCK_M 128 \
    --BLOCK_N 128 \
    --BLOCK_K 32 \
    --GROUP_M 4 \
    --num-warps 8 \
    --num-stages 3
```

结果记录：

- config：`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=4, num_warps=8, num_stages=3`
- smoke case：`kernel_ms.median = 0.022528 ms`，correctness 通过
- 主 benchmark：`kernel_ms.median = 0.156672 ms`
- 对 Iteration 1 最优结果改善：`1.63x`（`0.254976 -> 0.156672 ms`）
- 对 baseline 改善：`2.64x`（`0.413696 -> 0.156672 ms`）

`ncu` 关键指标变化：

- Triton kernel `Duration`：`446.98 us -> 234.66 us`
- `Memory Throughput`：`31.64 GB/s -> 58.08 GB/s`
- `Compute (SM) Throughput`：`33.34% -> 63.89%`
- `Registers Per Thread`：`236 -> 186`
- `No Eligible`：`61.96% -> 27.06%`
- `Eligible Warps Per Scheduler`：`0.53 -> 1.29`
- `Achieved Occupancy`：基本不变，仍约 `16.66%`
- `L2 Hit Rate`：基本持平，`88.29% -> 88.25%`

分析：

- 这轮收益非常明确，说明旧 kernel 的主要问题不是 DRAM 带宽不足，而是权重访问布局和 Triton matmul 读取模式不匹配
- 把权重预打包成 `KxN contiguous` 后，shared-memory / register 使用形态明显改善，直接反映为更低的寄存器压力和更高的可发射 warp 数
- occupancy 几乎没变，但 kernel 仍显著加速，进一步证明“先把每个 active warp 变得更容易发射”比盲目追求 occupancy 更重要
- 新 `ncu` 目标 kernel section 里已经不再出现旧版那种显式的 shared-load bank-conflict 提示，说明这轮布局修正至少大幅缓解了原来的 shared 访问问题

下一轮准备：

- 在 packed layout 基础上重新 sweep `num_warps / num_stages / GROUP_M`
- 再决定是否要进一步试 `BLOCK_K=64` 或更激进的 tile
- 把 `iter_03_packed_layout` 补成正式 archive，避免源码、profile 和文字记录继续漂移

## Iteration 6：在 packed layout 上复扫参数

目标：

- 验证旧实现下得到的 `w8/s3` 最优点，在新布局下是否仍然成立
- 判断是否需要把 `profile_target` 从 `s3` 切到 `s4`

云端执行命令：

```bash
python3 ./scripts/triton_perf_sweep.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --warmup 10 \
  --repeat 50 \
  --out perf/runs/triton_iterations/iter_04_packed_resweep
```

对最接近的两个候选补单点复测：

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source profile_target \
  --warmup 10 \
  --repeat 50 \
  --BLOCK_M 128 \
  --BLOCK_N 128 \
  --BLOCK_K 32 \
  --GROUP_M 4 \
  --num-warps 8 \
  --num-stages 3 \
  --json-output perf/runs/triton_iterations/iter_04_recheck_w8s3.json
```

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source profile_target \
  --warmup 10 \
  --repeat 50 \
  --BLOCK_M 128 \
  --BLOCK_N 128 \
  --BLOCK_K 32 \
  --GROUP_M 4 \
  --num-warps 8 \
  --num-stages 4 \
  --json-output perf/runs/triton_iterations/iter_04_recheck_w8s4.json
```

结果记录：

- full sweep best config：`bm128_bn128_bk32_gm4_w8_s4`
- full sweep best `kernel_ms.median`：`0.159648 ms`
- 同轮 sweep 中 `bm128_bn128_bk32_gm4_w8_s3`：`0.159744 ms`
- 单点复测：
  - `w8/s3 = 0.159744 ms`
  - `w8/s4 = 0.159744 ms`
- 两者差距已经低于正常 benchmark 抖动，且都没有稳定优于 Iteration 5 的 `0.156672 ms`

分析：

- 新布局下，最优区域仍然是 `128x128x32 + 8 warps`
- `num_stages=4` 在 sweep 中只给出了极小幅度、不可复现的名义优势，单点复测后与 `s3` 实际持平
- 说明当前最大的收益已经来自布局修正，而不是继续增加 pipeline stage
- 因此这轮不更新 `perf/configs/triton_linear_relu_a10.json`，继续保留 `profile_target = bm128_bn128_bk32_gm4_w8_s3`

下一轮准备：

- 如果继续深挖，优先考虑对当前 packed kernel 再做一轮新的 `ncu`
- 重点观察是否还能进一步压寄存器，或者在不伤害当前吞吐的前提下提高 active warps

## 阶段总结

最终需要补齐：

- Triton 相对 `generated_nvvm` 的提升
- Triton 相对 `cublas` 的距离
- 这组 shape-specialized config 是否值得接回 backend selection

当前阶段结论：

- 已在 A10 上跑通真实 Triton baseline / sweep / `nsys` / `ncu`
- 当前最优配置已经更新到 [`../configs/triton_linear_relu_a10.json`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/configs/triton_linear_relu_a10.json)
- 当前源码里已经落地一轮基于 `ncu` 的结构性优化：
  - 入口文件：[`../../scripts/triton_linear_relu_bench.py`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/scripts/triton_linear_relu_bench.py)
  - 核心手法：把 `Linear.weight` 预打包为 `KxN contiguous`
  - 当前最佳 standalone benchmark：`0.156672 ms`
- packed layout 之后的第一轮复扫已经完成：
  - sweep 名义最优是 `bm128_bn128_bk32_gm4_w8_s4`
  - 但单点复测显示 `w8/s3` 与 `w8/s4` 持平
  - 因此当前默认 profile 配置保持不变
- 最新 `ncu` 对 packed kernel 的主要观察：
  - `Registers Per Thread = 186`
  - `Achieved Occupancy = 16.66%`
  - `No Eligible = 27.06%`
  - `Eligible Warps Per Scheduler = 1.29`
  - `Memory Throughput = 58.08 GB/s`
  - `Compute (SM) Throughput = 63.89%`
  - `L2 Hit Rate = 88.25%`

下一轮建议：

- 在 packed layout 上重新做一轮 sweep，而不是继续沿用旧实现下得到的参数最优点
- 如果新 sweep 出现更优 `num_warps / num_stages / GROUP_M`，再补一轮 `nsys + ncu`
- `triton_profile_iter.py` 的 `ncu` NVTX filter 仍有技术债，短期内继续用 `perf_profile_ncu.sh` 作为 workaround

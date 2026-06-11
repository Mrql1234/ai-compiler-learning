# Triton A10 云端执行清单

本文档专门记录：由于本地没有 GPU，要在云端 A10 环境完成哪些事情，才能把 `compiler-mlir` 的 Triton 前 4 个里程碑真正跑起来。

适用范围：

- Triton 微基线
- Triton 参数 sweep
- Triton profile
- `fused_linear_relu` 多轮优化记录

## 1. 云端环境准备

### 1.1 基础软件

云端机器至少需要：

- NVIDIA A10
- 可用的 CUDA 驱动
- Python 3
- `torch` CUDA 版本
- `triton`
- `nsys`
- `ncu`

建议先确认：

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "import triton; print(triton.__version__)"
which nsys
which ncu
```

### 1.2 Nsight 权限

如果要跑 `ncu`，还需要确认 GPU performance counters 权限：

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

期望输出：

```text
RmProfilingAdminOnly: 0
```

如果不是 `0`，需要云端环境先解决 profiling 权限。

如果暂时拿不到管理员权限，可以先继续完成：

- Triton benchmark
- Triton sweep
- `nsys` 时间线

这时请显式使用：

```bash
PROFILE_SKIP_NCU=1 ./scripts/triton_cloud_a10_workflow.sh all
```

或只跑 profile：

```bash
PROFILE_SKIP_NCU=1 ./scripts/triton_cloud_a10_workflow.sh profile
```

这会保留 `bench + nsys`，跳过 `ncu` 相关产物。

## 2. 代码与目录准备

进入项目目录：

```bash
cd ~/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir
```

建议先确认这几个文件已经存在：

```bash
ls scripts/triton_cloud_a10_workflow.sh
ls scripts/triton_linear_relu_bench.py
ls scripts/triton_perf_sweep.py
ls scripts/triton_profile_iter.py
ls perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json
ls perf/configs/triton_linear_relu_a10.json
ls perf/notes/triton_linear_relu_iterations.md
```

## 2.1 推荐优先使用一键工作流入口

如果云端环境已经准备好，推荐优先用统一入口，而不是手工逐条执行：

```bash
./scripts/triton_cloud_a10_workflow.sh preflight
./scripts/triton_cloud_a10_workflow.sh all
```

这个脚本会按顺序执行：

- `preflight`
- `smoke`
- `baseline`
- `sweep`
- `profile`

如果只想单独跑某一阶段：

```bash
./scripts/triton_cloud_a10_workflow.sh smoke
./scripts/triton_cloud_a10_workflow.sh baseline
./scripts/triton_cloud_a10_workflow.sh sweep
./scripts/triton_cloud_a10_workflow.sh profile
```

常见覆盖方式是通过环境变量调整输出目录或 profile tag：

```bash
PROFILE_TAG=iter_03_group_m \
PROFILE_OUTPUT=perf/profiles/triton_iterations/iter_03_group_m \
./scripts/triton_cloud_a10_workflow.sh profile
```

如果只是想在本地或新机器上先看命令拼装是否正确，可以：

```bash
DRY_RUN=1 ./scripts/triton_cloud_a10_workflow.sh all
```

下面保留手工命令，便于单独调试每一步。

## 3. 里程碑 1：Triton 微基线

先跑小 case：

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m128_n128_k128.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source default \
  --warmup 5 \
  --repeat 20 \
  --json-output perf/runs/triton_iterations/smoke_m128.json
```

再跑主 case：

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source default \
  --warmup 10 \
  --repeat 50 \
  --json-output perf/runs/triton_iterations/iter_00_baseline.json
```

云端需要记录：

- correctness 是否通过
- `kernel_ms.median`
- `invoke_ms.median`
- `effective_gflops`
- `effective_gbps`

## 4. 里程碑 2：case / config 规范

这一阶段主要不是写代码，而是确认云端运行遵守同一份 case/config 合同。

云端需要做的事：

- 不要直接在脚本里改 shape
- shape 改动放进 `perf/cases/*.json`
- 参数改动放进 `perf/configs/*.json`
- 输出统一落到 `perf/runs/triton_iterations/`

## 5. 里程碑 3：benchmark + profile 工作流

### 5.1 sweep

```bash
python3 ./scripts/triton_perf_sweep.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --warmup 10 \
  --repeat 50 \
  --out perf/runs/triton_iterations/iter_01_tile
```

云端需要保存：

- `sweep_summary.json`
- `sweep_ranking.md`
- `best_config.json`
- 每个 candidate 的 benchmark JSON

### 5.2 profile

```bash
python3 ./scripts/triton_profile_iter.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source profile_target \
  --emit-nvtx \
  --tag iter_02_pipeline \
  --out perf/profiles/triton_iterations/iter_02_pipeline
```

云端需要保存：

- `*_profile_plan.json`
- `*.nsys-rep`
- `*.ncu-rep`
- `*_nsys_nvtx_summary.txt`
- `*_ncu_details.txt`
- `*_ncu_session.txt`

如果 `triton_profile_iter.py` 这一路因为 NVTX include filter 问题出现 `No kernels were profiled`，当前建议直接使用 full `ncu` workaround：

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

当前这轮 full `ncu` 的关键产物是：

- `perf/profiles/triton_iterations/iter_03_packed_layout/iter_03_packed_layout_ncu_full.ncu-rep`
- `perf/profiles/triton_iterations/iter_03_packed_layout/iter_03_packed_layout_ncu_full_details.txt`

## 6. 里程碑 4：多轮优化记录

云端每完成一轮，都要同步更新：

- `perf/notes/triton_linear_relu_iterations.md`

每轮至少补：

- 用了什么 config
- `kernel_ms` 如何变化
- occupancy / throughput / stall 怎么变化
- 为什么判断这轮优化有效或无效
- 下一轮要改什么

建议节奏：

1. Iteration 0：baseline
2. Iteration 1：tile sweep
3. Iteration 2：`num_warps` / `num_stages`
4. Iteration 3：`GROUP_M` / cache 行为
5. Iteration 4：epilogue 融合 / 写回路径
6. Iteration 5：权重预打包 / 访问布局修正

当前已验证的 Iteration 5 主 benchmark 命令：

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

当前已知结果：

- correctness：通过
- `kernel_ms.median = 0.156672 ms`
- 对旧最优 `0.254976 ms` 提升约 `1.63x`
- 对 baseline `0.413696 ms` 提升约 `2.64x`

## 7. 建议回传仓库的产物

推荐至少回传这些文件到仓库：

- `perf/runs/triton_iterations/iter_00_baseline.json`
- `perf/runs/triton_iterations/iter_01_tile/sweep_summary.json`
- `perf/runs/triton_iterations/iter_01_tile/sweep_ranking.md`
- `perf/profiles/triton_iterations/iter_02_pipeline/`
- `perf/notes/triton_linear_relu_iterations.md`
- `perf/archive/triton_iterations/iter_00_baseline/`
- `perf/archive/triton_iterations/iter_01_tile/`
- `perf/archive/triton_iterations/iter_02_pipeline/`
- `perf/profiles/triton_iterations/iter_03_packed_layout/`
- `perf/archive/triton_iterations/iter_03_packed_layout/`

建议在每轮 benchmark / sweep / profile 完成后，立即补一份结构化归档：

```bash
python3 ./scripts/triton_archive_iteration.py \
  --kind baseline \
  --iteration-name iter_00_baseline \
  --bench-json perf/runs/triton_iterations/iter_00_baseline.json
```

```bash
python3 ./scripts/triton_archive_iteration.py \
  --kind sweep \
  --iteration-name iter_01_tile \
  --sweep-summary perf/runs/triton_iterations/iter_01_tile/sweep_summary.json \
  --best-config perf/runs/triton_iterations/iter_01_tile/best_config.json
```

```bash
python3 ./scripts/triton_archive_iteration.py \
  --kind profile \
  --iteration-name iter_02_pipeline \
  --profile-plan perf/profiles/triton_iterations/iter_02_pipeline/iter_02_pipeline_profile_plan.json \
  --profile-dir perf/profiles/triton_iterations/iter_02_pipeline \
  --reference-bench-json perf/runs/triton_iterations/iter_01_tile/115_bm128_bn128_bk32_gm4_w8_s3.json
```

这样每一轮都会留下：

- `manifest.json`
- `metrics_summary.json`
- `analysis.md`
- `source_snapshot/`

## 8. 本地与云端分工

本地适合做：

- 脚本开发
- case/config 维护
- 结果整理
- notes 归档

云端适合做：

- 真实 benchmark
- Nsight profile
- 参数 sweep
- 真实优化记录

这份文档的作用就是把“哪些事情必须去云端做”单独固定下来，避免后面再混在主 README 里。

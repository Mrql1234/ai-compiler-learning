# compiler-mlir 性能入口

`perf/` 目录用于保存性能 case、运行结果、Nsight 报告和后续 Triton 迭代产物。

这次文档调整后，性能工作的重点不再是长期维护“`mlir_nvvm` / 手写 CUDA / `cublas` 三路线并行对比”，而是：

- 以 Triton 作为下一阶段主优化路线
- 先围绕 `fused_linear_relu` 做连续多轮迭代
- 再把同一套方法迁移到 `matmul`

详细设计请先看：

- [`PERF_MONITORING_PLAN.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/PERF_MONITORING_PLAN.md)
- [`TRITON_PERF_TASKS.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/TRITON_PERF_TASKS.md)
- [`LARGE_MODEL_GPU_DESIGN.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/LARGE_MODEL_GPU_DESIGN.md)
- [`LOWERING_ROADMAP.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/LOWERING_ROADMAP.md)
- [`CLOUD_TRITON_A10_WORKFLOW.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/CLOUD_TRITON_A10_WORKFLOW.md)

## 当前状态

当前 `compiler-mlir` 里已经可执行的 GPU 参考路线仍然是：

- `generated_nvvm` / `mlir_nvvm`
- `cublas`
- `cuda_hand`

其中：

- `generated_nvvm` 用于通用 MLIR GPU baseline
- `cublas` 用于库参考基线
- `cuda_hand` 只保留为历史教学/对照实现，不再作为新的长期主线

需要明确的是：

- `compiler-mlir` 的 Triton backend 集成仍然是**下一阶段设计目标**
- 但 `perf/` 里的 Triton `fused_linear_relu` 微基线已经是当前活跃优化主线，并且已有真实 A10 benchmark / `nsys` / `ncu` 数据

当前 Triton `linear_relu` 最新状态：

- 入口文件：`scripts/triton_linear_relu_bench.py`
- 当前实现已在 bench 脚本内部把 `Linear.weight` 预打包为 `KxN contiguous`
- 当前已知最佳 standalone 结果：
  - case：`perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json`
  - config：`bm128_bn128_bk32_gm4_w8_s3`
  - `kernel_ms.median = 0.156672 ms`
- packed layout 之后已经额外做过一轮复扫：
  - sweep 名义最优：`bm128_bn128_bk32_gm4_w8_s4 = 0.159648 ms`
  - 单点复测：`w8/s3` 与 `w8/s4` 都是 `0.159744 ms`
  - 当前不更新默认 config，继续保留 `w8/s3`
- 详细迭代分析见：
  - [`notes/triton_linear_relu_iterations.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/notes/triton_linear_relu_iterations.md)

## 入口文件

- `perf/cases/gpu_runner_demo.json`
  - 当前小型参考 case
- `perf/cases/linear_relu_f32_m1024_n1024_k1024.json`
  - 当前大 shape 参考 case
- `scripts/perf_run.py`
  - 当前统一运行入口
- `scripts/perf_compare.py`
  - 当前结果汇总入口
- `scripts/perf_profile_nsys.sh`
  - Nsight Systems 入口
- `scripts/perf_profile_ncu.sh`
  - Nsight Compute 入口
- `scripts/triton_linear_relu_bench.py`
  - Triton `fused_linear_relu` 微基线入口
- `scripts/triton_perf_sweep.py`
  - Triton 参数 sweep 入口
- `scripts/triton_profile_iter.py`
  - Triton profile 包装入口
- `scripts/triton_cloud_a10_workflow.sh`
  - 云端 A10 的 Triton 迭代工作流入口
- `scripts/triton_archive_iteration.py`
  - Triton 每轮归档入口，生成 manifest / metrics summary / analysis / source snapshot
- `perf/cases/triton_linear_relu_f32_m128_n128_k128.json`
  - Triton 小型 smoke case
- `perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json`
  - Triton 主 benchmark case
- `perf/configs/triton_linear_relu_a10.json`
  - Triton A10 配置
- `perf/configs/README.md`
  - Triton config 目录说明
- `perf/notes/README.md`
  - Triton 迭代记录目录说明
- `perf/notes/triton_linear_relu_iterations.md`
  - Triton `fused_linear_relu` 迭代记录模板
- `perf/CLOUD_TRITON_A10_WORKFLOW.md`
  - 云端 A10 必做事项与命令

## 当前可执行命令

构建：

```bash
cmake -S . -B build \
  -G Ninja \
  -DLLVM_DIR=/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/llvm \
  -DMLIR_DIR=/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/mlir \
  -DMINI_CUDA_ARCHITECTURES=86
cmake --build build -j2
```

查看当前 GPU lowering：

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
```

运行当前 `generated_nvvm` 基线：

```bash
./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
  --kernel-backend=generated_nvvm \
  --warmup=10 \
  --repeat=50
```

运行当前参考对比：

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cublas \
  --metric kernel_ms \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/gpu_runner_demo_reference
```

```bash
python3 ./scripts/perf_compare.py \
  --metric kernel_ms \
  perf/runs/gpu_runner_demo_reference/summary.json
```

采集 Nsight Systems：

```bash
./scripts/perf_profile_nsys.sh \
  perf/profiles/gpu_runner_demo_mlir_nvvm_nsys \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=1 \
    --repeat=2 \
    --cubin-format=fatbin
```

## Triton 新入口

云端 A10 一键工作流：

```bash
./scripts/triton_cloud_a10_workflow.sh preflight
./scripts/triton_cloud_a10_workflow.sh all
```

如果 `ncu` 因云端权限限制不可用，可以先保留 benchmark 和 `nsys`：

```bash
PROFILE_SKIP_NCU=1 ./scripts/triton_cloud_a10_workflow.sh all
```

如果只想单独执行某一阶段：

```bash
./scripts/triton_cloud_a10_workflow.sh smoke
./scripts/triton_cloud_a10_workflow.sh baseline
./scripts/triton_cloud_a10_workflow.sh sweep
./scripts/triton_cloud_a10_workflow.sh profile
```

常见覆盖参数通过环境变量传入，例如：

```bash
PROFILE_TAG=iter_03_group_m PROFILE_OUTPUT=perf/profiles/triton_iterations/iter_03_group_m \
./scripts/triton_cloud_a10_workflow.sh profile
```

Triton 微基线：

```bash
python3 ./scripts/triton_linear_relu_bench.py \
  --case perf/cases/triton_linear_relu_f32_m128_n128_k128.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source default \
  --warmup 5 \
  --repeat 20 \
  --json-output perf/runs/triton_iterations/smoke_m128.json
```

当前推荐的主 benchmark 命令：

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

Triton sweep：

```bash
python3 ./scripts/triton_perf_sweep.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --warmup 10 \
  --repeat 50 \
  --out perf/runs/triton_iterations/iter_01_tile
```

Triton profile：

```bash
python3 ./scripts/triton_profile_iter.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --config-source profile_target \
  --emit-nvtx \
  --tag iter_02_pipeline \
  --out perf/profiles/triton_iterations/iter_02_pipeline
```

如果 `triton_profile_iter.py` 的 `ncu` NVTX 过滤没有抓到 kernel，当前推荐直接走 full `ncu` workaround：

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

Triton 迭代归档：

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

```bash
python3 ./scripts/triton_archive_iteration.py \
  --kind profile \
  --iteration-name iter_03_packed_layout \
  --profile-plan perf/profiles/triton_iterations/iter_03_packed_layout/iter_03_packed_layout_profile_plan.json \
  --profile-dir perf/profiles/triton_iterations/iter_03_packed_layout \
  --reference-bench-json perf/runs/triton_iterations/iter_03_packed_bench.json
```

```bash
python3 ./scripts/triton_archive_iteration.py \
  --kind sweep \
  --iteration-name iter_04_packed_resweep \
  --sweep-summary perf/runs/triton_iterations/iter_04_packed_resweep/sweep_summary.json \
  --best-config perf/runs/triton_iterations/iter_04_packed_resweep/best_config.json
```

归档结果默认落到：

- `perf/archive/triton_iterations/<iteration>/manifest.json`
- `perf/archive/triton_iterations/<iteration>/metrics_summary.json`
- `perf/archive/triton_iterations/<iteration>/analysis.md`
- `perf/archive/triton_iterations/<iteration>/source_snapshot/`

## 已归档数据

当前仓库已保留一批 A10 真实运行结果，主要用于：

- 回看旧的参考基线
- 对照后续 Triton 方案的阶段性提升

主要归档包括：

- `perf/runs/gpu_runner_demo_a10_20260511/`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/`
- `perf/profiles/a10_20260511/`
- `perf/archive/triton_iterations/iter_00_baseline/`
- `perf/archive/triton_iterations/iter_01_tile/`
- `perf/archive/triton_iterations/iter_02_pipeline/`
- `perf/archive/triton_iterations/iter_03_packed_layout/`
- `perf/archive/triton_iterations/iter_04_packed_resweep/`

## 下一阶段应新增的内容

按新的 Triton 主线，当前已经落地了：

- Triton 微基线运行入口
- Triton 参数 sweep 入口
- Triton profile 归档入口
- Triton A10 case/config/notes/workflow 文档
- 云端 A10 一键执行脚本

后续还建议继续补齐：

- packed layout 基础上的新一轮 sweep / profile 结果
- `matmul` 的 Triton case / config / notes
- 迭代完成后把成熟配置接回 backend selection
- 更进一步的 Triton backend 集成能力

其中真实 benchmark / profile 数据仍然需要在云端 GPU 机器上运行后再回传仓库。

如果要按任务顺序推进，请直接看：

- [`TRITON_PERF_TASKS.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/TRITON_PERF_TASKS.md)

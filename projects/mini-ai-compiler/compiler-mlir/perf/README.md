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

- `compiler-mlir` 的 Triton backend 目前还是**下一阶段设计目标**
- 现有 `perf/` 里的归档结果主要还是旧的三路线快照

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

## 已归档数据

当前仓库已保留一批 A10 真实运行结果，主要用于：

- 回看旧的参考基线
- 对照后续 Triton 方案的阶段性提升

主要归档包括：

- `perf/runs/gpu_runner_demo_a10_20260511/`
- `perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/`
- `perf/profiles/a10_20260511/`

## 下一阶段应新增的内容

按新的 Triton 主线，后续建议补齐：

- Triton 微基线运行入口
- Triton 参数 sweep 入口
- Triton profile 归档目录
- Triton config 文件
- `fused_linear_relu` 与 `matmul` 的 A10 专项 case

这些内容目前还没有在 `compiler-mlir` 中实装，后续实现时应同步更新本目录 README。

如果要按任务顺序推进，请直接看：

- [`TRITON_PERF_TASKS.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/TRITON_PERF_TASKS.md)

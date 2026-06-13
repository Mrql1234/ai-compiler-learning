# compiler-mlir

This directory hosts the formal MLIR-native compiler track for `mini-ai-compiler`.

## Purpose

- Register the `mini` dialect skeleton
- Host canonicalization / folding / DCE / fusion pass skeletons
- Provide a compiler driver similar to `mlir-opt`
- Become the main lowering pipeline for:
  - `MLIR -> LLVM IR -> CPU`
  - `MLIR -> Triton/GPU`

## Current Scope

The current version has moved beyond a pure skeleton:

It can now:

- compile against an existing LLVM/MLIR build
- link against the local CUDA driver toolchain for GPU execution
- parse and verify `mini.constant`, `mini.linear`, `mini.matmul`, `mini.add`, `mini.relu`, `mini.fused_linear_relu`, `mini.fused_matmul_add_relu`
- run mini canonicalization / fusion / constant-fold passes
- lower `mini.*` ops to `linalg` / `arith` / `tensor`
- continue into an experimental bufferized CPU-oriented path with standard MLIR passes
- lower `mini.*` ops into `gpu.launch_func` / `gpu.module`
- lower the GPU path further into NVVM binaries
- JIT-run a small lowered GPU demo locally and return the computed result
- collect repeatable GPU performance runs for compiler-generated, legacy hand
  CUDA, and third-party library kernel baselines while preparing for a
  Triton-first iteration flow
- run a first teaching-style weight-only INT8 quantization pass for `mini.linear` / `mini.fused_linear_relu`

## Configure

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

Notes:

- `MLIR_DIR` and `LLVM_DIR` should come from the same LLVM/MLIR build
- a working CUDA driver toolkit is now required for the local GPU runner path

## Design Notes

- Multi-backend lowering roadmap:
  - `LOWERING_ROADMAP.md`
- 大模型算子与 Triton/GPU 路线设计:
  - `LARGE_MODEL_GPU_DESIGN.md`
- Local dev + cloud A10 workflow:
  - `GPU_A10_WORKFLOW.md`
- GPU performance monitoring plan:
  - `PERF_MONITORING_PLAN.md`

## Useful Commands

Lower mini ops to standard tensor/linalg dialects:

```bash
./build/bin/mini-compiler-opt --mini-lower-to-linalg test/lower_to_linalg.mlir
```

Quantize supported constant linear weights to INT8 first:

```bash
./build/bin/mini-compiler-opt --mini-quantize-weights test/quantize_weights.mlir
```

Inspect the late-staged `mini.qlinear` form before the dedicated quantized GPU lowering:

```bash
./build/bin/mini-compiler-opt \
  --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion,mini-quantize-weights,mini-lower-to-linalg))' \
  test/quantized_qlinear_late_stage.mlir
```

Prepare the quantized GPU path so dequantization is fused into the qlinear
matmul-style `linalg.generic` body:

```bash
./build/bin/mini-compiler-opt --mini-quantized-gpu-prep test/quantized_lower_to_linalg.mlir
```

Continue one step further into bufferized IR:

```bash
./build/bin/mini-compiler-opt \
  --mini-lower-to-linalg \
  --one-shot-bufferize \
  test/lower_to_bufferized.mlir
```

Continue all the way to LLVM dialect on the CPU path:

```bash
./build/bin/mini-compiler-opt --mini-cpu-lowering test/cpu_pipeline.mlir
./build/bin/mini-compiler-opt --mini-quantized-cpu-lowering test/quantized_lower_to_linalg.mlir
```

Run a lowered MLIR module through the local CPU JIT runner:

```bash
./build/bin/mini-compiler-runner test/cpu_runner_demo.mlir --entry-point-result=f32
./build/bin/mini-compiler-runner --quantized test/quantized_runner_demo.mlir --entry-point-result=f32
```

Prepare `mini.*` programs for a later GPU/Triton route:

```bash
./build/bin/mini-compiler-opt --mini-gpu-prep test/gpu_prep.mlir
./build/bin/mini-compiler-opt --mini-quantized-gpu-prep test/quantized_lower_to_linalg.mlir
```

Lower further into GPU launch/module form without needing a local GPU:

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
./build/bin/mini-compiler-opt --mini-quantized-gpu-lowering test/quantized_gpu_lowering.mlir
```

The quantized GPU lowering keeps the INT8 weight load inside the generated
matmul kernel body, where the kernel performs `arith.sitofp`, multiplies by the
weight scale, and accumulates into the output.

Inspect the project-defined GPU loop mapping strategy on `scf.parallel`:

```bash
./build/bin/mini-compiler-opt test/gpu_map.mlir --mini-gpu-map
```

Inspect the current default tiling + mapping strategy:

```bash
./build/bin/mini-compiler-opt test/gpu_tile_map.mlir --mini-gpu-tile --mini-gpu-map
```

Current default tile sizes are `8 x 8` for the leading 2 GPU dimensions.

Override tile sizes explicitly for the standalone tiling stage:

```bash
./build/bin/mini-compiler-opt test/gpu_tile_options.mlir --mini-gpu-tile-pipeline="tile-sizes=4,2"
```

Or override the tile sizes inside the full GPU lowering pipeline:

```bash
./build/bin/mini-compiler-opt test/gpu_prep.mlir --mini-gpu-lowering="tile-sizes=16,8"
```

The current memory pass now does two things for non-`gpu.alloc` launch operands:

- materialize a `gpu.alloc host_shared` buffer before `gpu.launch_func`
- skip copy-back for read-only sources such as constant `memref.global`
- conservatively insert a copy-back after the launch for mutable operands

You can inspect that behavior with:

```bash
./build/bin/mini-compiler-opt test/gpu_host_shared_copyback.mlir --mini-gpu-host-shared
./build/bin/mini-compiler-opt test/gpu_host_shared_readonly.mlir --mini-gpu-host-shared
```

Run the local GPU JIT demo through the new compiler-mlir GPU runner:

```bash
./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir
./build/bin/mini-compiler-gpu-runner --quantized test/quantized_runner_demo.mlir
```

Expected output:

```text
3.500000e+00
```

Run the GPU runner with internal warmup/repeat timing, JSON output, and a
lowered MLIR artifact:

```bash
./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
  --warmup=10 \
  --repeat=100 \
  --json-output=perf/runs/gpu_runner_demo_mlir_nvvm.json \
  --dump-lowered=perf/runs/gpu_runner_demo_lowered.mlir \
  --cubin-format=fatbin
```

Preflight the cloud A10 NVVM toolchain:

```bash
./scripts/a10_preflight.sh
```

Run the staged A10 NVVM lowering pipeline:

```bash
./scripts/a10_lower_to_nvvm.sh test/gpu_prep.mlir
```

Run a minimal correctness + performance comparison between the CPU and GPU
runner paths:

```bash
python3 ./scripts/benchmark_compare.py test/gpu_runner_demo.mlir \
  --warmup 1 \
  --repeat 5
```

This harness:

- runs the CPU runner and GPU runner on the same MLIR module
- checks the final numeric result with configurable tolerances
- reports average / median / min / max latency
- prints a simple CPU-vs-GPU speedup summary

Useful options:

```bash
python3 ./scripts/benchmark_compare.py test/gpu_runner_demo.mlir \
  --entry-function=run \
  --result-type=f32 \
  --warmup 2 \
  --repeat 10 \
  --gpu-extra-arg=--gpu-chip=sm_86 \
  --gpu-extra-arg=--cubin-format=fatbin
```

If you only want a CPU baseline on a machine without a working NVPTX/CUDA path:

```bash
python3 ./scripts/benchmark_compare.py test/gpu_runner_demo.mlir --skip-gpu
```

GPU 性能与 Triton 迭代入口：

- `PERF_MONITORING_PLAN.md`
  - 新的 GPU 性能主文档
  - 已改为以 Triton 为主线的性能迭代方案
- `TRITON_PERF_TASKS.md`
  - 从设计到实现的任务拆解文档
- `LARGE_MODEL_GPU_DESIGN.md`
  - Triton 在整体 GPU 后端体系里的角色说明
- `LOWERING_ROADMAP.md`
  - `mini -> 标准 dialect -> backend split` 的分层路线
- `perf/README.md`
  - `perf/` 目录的快速入口

这次调整后的设计结论是：

- 后续性能工作不再以手写 CUDA 作为长期主线
- `compiler-mlir` 的下一阶段主优化路线应改为 Triton
- 第一阶段优先做 `fused_linear_relu` 的 Triton 迭代优化
- 第二阶段再把同样的方法迁移到 `matmul`
- `generated_nvvm` 和 `cublas` 只保留为阶段性参考基线

需要明确的是：

- 当前 `compiler-mlir` 里真正可执行的 GPU 参考路线仍然是 `generated_nvvm` / `mlir_nvvm`、`cublas`、`cuda_hand`
- Triton backend 在 `compiler-mlir` 中还属于下一阶段实现目标
- 因此新的 `PERF_MONITORING_PLAN.md` 描述的是“下一阶段性能迭代设计”，不是已经全部落地的能力清单

当前建议从下面这些入口继续推进：

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir

./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
  --kernel-backend=generated_nvvm \
  --warmup=10 \
  --repeat=50

python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cublas \
  --metric kernel_ms \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/gpu_runner_demo_reference

python3 ./scripts/perf_compare.py \
  --metric kernel_ms \
  perf/runs/gpu_runner_demo_reference/summary.json
```

如果要看当前 `mini.fused_linear_relu` 的 runtime-call 分叉入口，可继续使用：

```bash
./build/bin/mini-compiler-opt test/gpu_runtime_call_lowering.mlir \
  --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion),mini-gpu-runtime-call-lowering{backend=cublas})'
```

如果要采集当前 baseline 的 Nsight Systems 报告，可使用：

```bash
./scripts/perf_profile_nsys.sh \
  perf/profiles/gpu_runner_demo_mlir_nvvm_nsys \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=1 \
    --repeat=2 \
    --cubin-format=fatbin
```

如果要继续推进 Triton 主线，推荐先按下面顺序阅读和执行：

1. `PERF_MONITORING_PLAN.md`
2. `TRITON_PERF_TASKS.md`
3. `perf/README.md`
4. `perf/configs/README.md`
5. `perf/notes/README.md`
6. `perf/CLOUD_TRITON_A10_WORKFLOW.md`

## Triton 算子 Agent 原型

为了把“算子需求解析 -> kernel 候选生成 -> benchmark / profiling -> 反馈驱动 resweep”做成统一入口，当前新增了一套 Triton-first Agent 原型。

入口文件：

- `scripts/triton_operator_agent.py`
- `scripts/triton_operator_agent_lib.py`
- `perf/specs/triton_agent_fused_linear_relu_a10.json`
- `perf/specs/triton_agent_matmul_a10.json`
- `perf/specs/triton_agent_softmax_a10.json`
- `perf/specs/triton_agent_layernorm_a10.json`

当前覆盖范围：

- `fused_linear_relu`
  - 已接入可执行的 Triton benchmark / profile 流程
  - 适合做完整闭环演示
- `matmul`、`softmax`、`layernorm`
  - 已接入统一规格描述、候选生成、经验记忆和 Nsight 诊断接口
  - 当前还是 planner-only 原型

常用命令：

```bash
python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode plan \
  --dry-run

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode tune \
  --dry-run \
  --max-candidates 4 \
  --max-iterations 1

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_matmul_a10.json \
  --mode plan \
  --dry-run
```

如果已经采集到了 Nsight Compute 文本，可以单独做瓶颈分析：

```bash
python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode analyze \
  --ncu-details /path/to/iter_best_ncu_details.txt \
  --run-dir perf/runs/agent_runs/analyze_linear_relu
```

Translate the LLVM dialect output into textual LLVM IR:

```bash
./build/bin/mini-compiler-opt --mini-cpu-lowering test/cpu_pipeline.mlir \
  | /path/to/matching/mlir-translate --mlir-to-llvmir
```

Important:

- `mlir-translate` should come from the **same LLVM/MLIR build** as the one used to build `mini-compiler-opt`
- mixing the custom source build with older system binaries can fail on newer LLVM dialect ops such as `llvm.mlir.poison`
- the local GPU runner currently relies on the project-provided CUDA runtime wrappers plus matching `mlir_runner_utils` libraries from the same LLVM build

## Expected Next Steps

1. extend the runnable GPU path beyond `test/gpu_runner_demo.mlir`
2. reduce the current host-shared bridge into a cleaner explicit device-memory lowering story
3. add more mini ops and MLIR-native optimization passes
4. connect the Python bridge to emit stable MLIR input for this toolchain
5. map the same high-level ops into a Triton-oriented backend path

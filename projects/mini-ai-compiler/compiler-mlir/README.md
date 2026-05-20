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
- collect repeatable GPU performance runs for compiler-generated, hand CUDA,
  and third-party library kernel baselines
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

GPU 性能监控入口文件：

- `PERF_MONITORING_PLAN.md`：完整 GPU 性能监控方案、入口文件和命令说明
- `perf/README.md`：`perf/` 目录的快速入口说明
- `perf/cases/gpu_runner_demo.json`：小型 demo case，包含 `mlir_nvvm`、`cuda_hand`、`cublas`
- `perf/cases/linear_relu_f32_m1024_n1024_k1024.json`：大型 `linear + relu` case
- `scripts/perf_run.py`：统一运行入口，生成 backend JSON 和 `summary.json`
- `scripts/perf_compare.py`：对比 `summary.json`
- `scripts/perf_profile_nsys.sh`：Nsight Systems 包装脚本
- `scripts/perf_profile_ncu.sh`：Nsight Compute 包装脚本
- `scripts/perf_validate_cloud.sh`：云 GPU 上的一键构建、运行三 backend、按 `kernel_ms` 对比的验证入口
- `lib/GpuPasses.cpp`：包含 `mini-gpu-runtime-call-lowering`，用于把 `mini.fused_linear_relu` 降到显式 GPU runtime `func.call`
- `test/gpu_runtime_call_lowering.mlir`：验证 `cuda_hand` / `cublas` runtime-call lowering 的 lit smoke test
- `runtime/MiniCudaKernelRuntime.cu`：提供 runner integrated path 和 executable memref runtime-call path 使用的 CUDA/cuBLAS ABI
- `tools/mini-compiler-runner.cpp`：支持 `--lowering-pipeline=...`，可直接执行 runtime-call lowering pipeline
- `tools/mini-compiler-kernel-bench.cpp`：手写 CUDA / cuBLAS benchmark 入口

性能监控方案的目标口径是对比三条路线最终 kernel 的 `kernel_ms`，不把 lowering、JIT engine 创建、输入构造、显存分配、H2D/D2H 拷贝计入主指标。`mlir_nvvm` 走编译器 lowering 后的 runner 路线；`cuda_hand` / `cublas` 现在也可由同一个 `mini-compiler-gpu-runner` 通过 backend selection 调用统一 runtime ABI。

当前 harness 已支持新的 metric schema：`mini-compiler-kernel-bench` 输出 CUDA event 口径的 `metrics.kernel_ms` 和 host 侧 `metrics.invoke_ms`；`mini-compiler-gpu-runner` 输出 `metrics.kernel_ms`、`metrics.invoke_ms`、`compile_ms`、`engine_create_ms`、`end_to_end_ms`，其中 `mlir_nvvm` 的 `kernel_ms` 来自 `mgpuLaunchKernel` 内的 CUDA driver event accumulator。`perf_compare.py` 默认按 `kernel_ms` 对比，查看旧归档 v0 数据时使用 `--metric latency_ms`。

`mini-compiler-gpu-runner` 现在提供稳定的 backend selection 入口：`--kernel-backend=generated_nvvm|mlir_nvvm|cuda_hand|cublas|cutlass`。当前 `generated_nvvm` / `mlir_nvvm`、`cuda_hand`、`cublas` 是可执行路线；`cutlass` 会返回明确的未实现错误。

编译器集成路线使用的 CUDA runtime ABI 位于 `runtime/MiniCudaKernelRuntime.cu`，当前提供 `mini_cuda_linear_relu_f32` 和 `mini_cublas_linear_relu_f32`。`mini-gpu-runtime-call-lowering` 已经能把静态 shape 的 `mini.fused_linear_relu` 降到 `mini_cuda_linear_relu_f32_memref` / `mini_cublas_linear_relu_f32_memref` 形式的显式 `func.call`，并可通过 `mini-gpu-runtime-call-lowering-pipeline` 降到 LLVM 后由 `mini-compiler-runner` 执行。

查看 runtime-call lowering IR：

```bash
./build/bin/mini-compiler-opt test/gpu_runtime_call_lowering.mlir \
  --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion),mini-gpu-runtime-call-lowering{backend=cuda_hand})'

./build/bin/mini-compiler-opt test/gpu_runtime_call_lowering.mlir \
  --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion),mini-gpu-runtime-call-lowering{backend=cublas})'
```

执行 runtime-call lowering pipeline：

```bash
./build/bin/mini-compiler-runner test/gpu_runner_demo.mlir \
  --lowering-pipeline='mini-gpu-runtime-call-lowering-pipeline{backend=cuda_hand}' \
  --shared-libs=build/lib/libMiniCudaRuntimeWrappers.so

./build/bin/mini-compiler-runner test/gpu_runner_demo.mlir \
  --lowering-pipeline='mini-gpu-runtime-call-lowering-pipeline{backend=cublas}' \
  --shared-libs=build/lib/libMiniCudaRuntimeWrappers.so
```

本地 CPU-only smoke check 可用 dummy external backend 验证 harness 本身：

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend cuda_hand \
  --backend-command cuda_hand='printf 3.5' \
  --metric invoke_ms \
  --warmup 1 \
  --repeat 2 \
  --run-dir /tmp/compiler-mlir-perf-smoke
python3 ./scripts/perf_compare.py /tmp/compiler-mlir-perf-smoke/summary.json
```

在 A10 云 GPU 上运行小型 demo 的三 backend 对比：

```bash
./scripts/perf_validate_cloud.sh
```

等价的手动命令：

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend cublas \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/gpu_runner_demo_a10_20260511
```

查看已归档的小型 demo 对比结果：

```bash
python3 ./scripts/perf_compare.py \
  --metric latency_ms \
  perf/runs/gpu_runner_demo_a10_20260511/summary.json
```

在 A10 云 GPU 上运行大型 `linear + relu` 的 CUDA/cuBLAS 对比：

```bash
python3 ./scripts/perf_run.py \
  perf/cases/linear_relu_f32_m1024_n1024_k1024.json \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511
```

查看已归档的大型 case 对比结果：

```bash
python3 ./scripts/perf_compare.py \
  --metric latency_ms \
  perf/runs/linear_relu_f32_m1024_n1024_k1024_a10_20260511/summary.json
```

采集 `mlir_nvvm` 路线的 Nsight Systems 报告：

```bash
./scripts/perf_profile_nsys.sh \
  perf/profiles/a10_20260511/gpu_runner_demo_mlir_nvvm_nsys \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=1 \
    --repeat=2 \
    --cubin-format=fatbin
```

采集手写 CUDA kernel 的 Nsight Compute 报告：

```bash
./scripts/perf_profile_ncu.sh \
  perf/profiles/a10_20260511/gpu_runner_demo_cuda_hand_ncu \
  ./build/bin/mini-compiler-kernel-bench \
    --backend cuda_hand \
    --case perf/cases/gpu_runner_demo.json \
    --warmup 1 \
    --repeat 1
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

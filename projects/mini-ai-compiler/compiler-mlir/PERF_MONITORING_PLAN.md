# compiler-mlir GPU Performance Monitoring Plan

This document describes the performance monitoring system for the
`compiler-mlir` GPU route. It is intentionally split into local, GPU-independent
work and cloud GPU work so development can continue on a CPU-only machine and
profiling can continue later on an A10 or another CUDA server.

## Goals

The performance system compares three kernel sources with the same case,
correctness rule, timing protocol, and report format:

- `mlir_nvvm`: compiler-generated kernels from `mini-compiler-gpu-runner`
- `cuda_hand`: hand-written CUDA kernels
- `cutlass` or another library backend: third-party kernels such as CUTLASS,
  cuBLAS, or cuDNN

The comparison should answer:

- whether the compiler-generated kernel is correct
- how far it is from a hand-written CUDA baseline
- how far it is from a third-party library baseline
- whether the bottleneck is launch overhead, memcpy, runtime overhead, or kernel
  execution
- which compiler pass, lowering choice, or runtime wrapper should be optimized

## Current GPU-Independent Pieces

These pieces can be developed and tested without local CUDA libraries:

- `perf/cases/gpu_runner_demo.json`: first shared perf case definition
- `scripts/perf_run.py`: backend dispatcher and unified JSON result writer
- `scripts/perf_compare.py`: summary comparison table
- `scripts/perf_profile_nsys.sh`: Nsight Systems wrapper
- `scripts/perf_profile_ncu.sh`: Nsight Compute wrapper
- `perf/README.md`: quick command reference

The GPU runner also has GPU-independent command-line plumbing for profiling:

- `--warmup=N`
- `--repeat=N`
- `--json-output=path`
- `--dump-lowered=path`
- `--ptxas-cmd-options=...`

## Architecture

The top-level flow is:

```text
perf case
  -> perf_run.py
     -> mlir_nvvm backend
        -> mini-compiler-gpu-runner
        -> mini -> gpu -> nvvm -> fatbin
        -> ExecutionEngine
     -> cuda_hand backend
        -> external benchmark command
        -> hand-written CUDA kernel
     -> cutlass backend
        -> external benchmark command
        -> CUTLASS/cuBLAS/cuDNN kernel
  -> per-backend JSON
  -> summary.json
  -> perf_compare.py
  -> Nsight artifacts when profiling is enabled
```

The shared contract is a case JSON plus one JSON output per backend. This keeps
the comparison layer independent from how each kernel is implemented.

## Case Contract

A case file should define:

- `name`: stable case identifier
- `input`: MLIR input used by `mlir_nvvm`
- `entry_function`: runner entry point
- `result_type`: currently `f32` or `void`
- `tolerance`: `abs` and `rel` correctness tolerance
- `backends`: backend definitions

Current example:

```bash
perf/cases/gpu_runner_demo.json
```

Future larger cases should be added alongside it:

```text
perf/cases/linear_relu_f32_m1024_n1024_k1024.json
perf/cases/matmul_f32_m2048_n2048_k2048.json
perf/cases/qlinear_int8_m1024_n1024_k1024.json
```

## Result Contract

Each backend result should provide:

- backend name
- backend kind
- command
- return code
- scalar result when available
- correctness status
- `timings_ms`
- latency summary: min, mean, median, max
- artifact paths, such as lowered MLIR or Nsight output

The run directory contains:

```text
perf/runs/<case>_<timestamp>/
  mlir_nvvm.json
  cuda_hand.json
  cutlass.json
  mlir_nvvm_lowered.mlir
  summary.json
```

## Local CPU-Only Workflow

Use this workflow on a machine without CUDA libraries to validate the harness
itself:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend cuda_hand \
  --backend-command cuda_hand='printf 3.5' \
  --warmup 1 \
  --repeat 2 \
  --run-dir /tmp/compiler-mlir-perf-smoke
```

Then compare the generated summary:

```bash
python3 ./scripts/perf_compare.py /tmp/compiler-mlir-perf-smoke/summary.json
```

This does not test GPU execution. It only verifies that external backend
dispatch, JSON writing, correctness comparison, and summary reporting work.

## Cloud GPU Workflow

On an A10 or another CUDA machine, first configure and build with CUDA support:

```bash
cmake -S . -B build \
  -G Ninja \
  -DLLVM_DIR=/home/ql/code/llvm_clang_static_analyzer/build-mlir/lib/cmake/llvm \
  -DMLIR_DIR=/home/ql/code/llvm_clang_static_analyzer/build-mlir/lib/cmake/mlir
cmake --build build -j2
```

Run the compiler-generated baseline:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --warmup 10 \
  --repeat 100
```

Run a hand CUDA backend once a benchmark binary exists:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend-command cuda_hand='./build/bin/mini-compiler-kernel-bench --backend cuda_hand --case perf/cases/gpu_runner_demo.json' \
  --warmup 10 \
  --repeat 100
```

Run a third-party backend once a CUTLASS/cuBLAS benchmark exists:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cutlass \
  --backend-command cutlass='./build/bin/mini-compiler-kernel-bench --backend cutlass --case perf/cases/gpu_runner_demo.json' \
  --warmup 10 \
  --repeat 100
```

Compare a run:

```bash
python3 ./scripts/perf_compare.py perf/runs/<run-dir>/summary.json
```

## Nsight Workflow

Use Nsight Systems to inspect runtime-level behavior:

```bash
./scripts/perf_profile_nsys.sh perf/runs/nsys_gpu_runner_demo \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=10 \
    --repeat=100 \
    --cubin-format=fatbin
```

Use Nsight Compute to inspect kernel-level metrics:

```bash
./scripts/perf_profile_ncu.sh perf/runs/ncu_gpu_runner_demo \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=5 \
    --repeat=20 \
    --cubin-format=fatbin
```

Nsight Systems should be used first to decide whether the bottleneck is:

- launch overhead
- host/device transfer
- CUDA synchronization
- runtime wrapper overhead
- actual kernel time

Nsight Compute should then be used to inspect:

- achieved occupancy
- register pressure
- memory throughput
- global load/store efficiency
- shared memory behavior
- warp stall reasons
- tensor core utilization for library or Tensor Core kernels

## Cloud Implementation Tasks

### Task 1: Stabilize `mlir_nvvm`

- Run `a10_preflight.sh`
- Run `mini-compiler-gpu-runner` with `--cubin-format=fatbin`
- Confirm JSON output is written
- Confirm `--dump-lowered` writes the lowered MLIR artifact
- Confirm `perf_run.py --backend mlir_nvvm` produces `summary.json`

### Task 2: Add Hand CUDA Backend

Create a future benchmark binary such as:

```text
tools/mini-compiler-kernel-bench.cpp
```

Required command shape:

```bash
./build/bin/mini-compiler-kernel-bench \
  --backend cuda_hand \
  --case perf/cases/gpu_runner_demo.json \
  --warmup 10 \
  --repeat 100
```

The first version may simply print the scalar result to stdout. A later version
should write the same JSON result contract as `perf_run.py`.

Recommended first kernel:

```text
linear + relu, f32, fixed small shape matching gpu_runner_demo
```

### Task 3: Add Third-Party Backend

Start with cuBLAS for matmul or CUTLASS for fused GEMM epilogues:

```bash
./build/bin/mini-compiler-kernel-bench \
  --backend cutlass \
  --case perf/cases/linear_relu_f32_m1024_n1024_k1024.json \
  --warmup 10 \
  --repeat 100
```

The third-party backend should use the same input values and tolerance as the
compiler-generated route.

### Task 4: Add NVTX Ranges

Add NVTX ranges in runner or runtime code for:

- `compile`
- `engine_create`
- `warmup`
- `benchmark`
- `kernel_launch`
- `copyback`
- `verify`

This makes Nsight Systems readable and connects runtime events back to the perf
case.

### Task 5: Add Larger Cases

After the small demo is stable, add cases large enough to expose meaningful GPU
behavior:

- `linear_relu_f32_m1024_n1024_k1024`
- `matmul_f32_m2048_n2048_k2048`
- `qlinear_int8_m1024_n1024_k1024`

Each case should include a compiler-generated backend, a hand CUDA backend, and
a third-party backend when available.

## Optimization Loop

Use this loop for every performance change:

```text
1. Run perf_run.py for all available backends.
2. Run perf_compare.py to quantify the gap.
3. Use Nsight Systems to classify runtime vs kernel bottlenecks.
4. Use Nsight Compute for kernel-level bottlenecks.
5. Update compiler passes, lowering options, hand CUDA code, or library config.
6. Re-run the same case.
7. Save the new summary under perf/baselines/<gpu>/ when it becomes a baseline.
```

The key metric is the gap between:

- `mlir_nvvm` and `cuda_hand`
- `mlir_nvvm` and `cutlass`

The hand CUDA baseline explains what a direct expert kernel can do. The
third-party backend shows the expected high-performance library ceiling. The
compiler-generated route is the main optimization target.

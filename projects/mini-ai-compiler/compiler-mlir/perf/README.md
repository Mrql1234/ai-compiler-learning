# compiler-mlir Performance Harness

This directory stores repeatable performance cases, run outputs, baselines, and
reports for comparing three GPU kernel sources:

- compiler-generated kernels from `mini-compiler-gpu-runner`
- hand-written CUDA kernels exposed through an external benchmark command
- third-party kernels such as CUTLASS or cuBLAS exposed through an external benchmark command

## Entry Files

- `perf/cases/gpu_runner_demo.json`: first runnable case definition
- `PERF_MONITORING_PLAN.md`: cloud GPU implementation and optimization plan
- `scripts/perf_run.py`: runs selected backends and writes comparable JSON
- `scripts/perf_compare.py`: prints a compact comparison table from `summary.json`
- `scripts/perf_profile_nsys.sh`: wraps a command with Nsight Systems
- `scripts/perf_profile_ncu.sh`: wraps a command with Nsight Compute

## Run Commands

Run a local CPU-only smoke check with a dummy external backend:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend cuda_hand \
  --backend-command cuda_hand='printf 3.5' \
  --warmup 1 \
  --repeat 2 \
  --run-dir /tmp/compiler-mlir-perf-smoke
python3 ./scripts/perf_compare.py /tmp/compiler-mlir-perf-smoke/summary.json
```

Run the enabled compiler-generated backend for the demo case on a CUDA-enabled
cloud GPU machine:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --warmup 10 \
  --repeat 100
```

Compare an existing run:

```bash
python3 ./scripts/perf_compare.py perf/runs/<run-dir>/summary.json
```

Add a hand CUDA backend once a benchmark binary exists:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend-command cuda_hand='./build/bin/mini-compiler-kernel-bench --backend cuda_hand --case perf/cases/gpu_runner_demo.json' \
  --warmup 10 \
  --repeat 100
```

Add a CUTLASS or cuBLAS backend the same way:

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cutlass \
  --backend-command cutlass='./build/bin/mini-compiler-kernel-bench --backend cutlass --case perf/cases/gpu_runner_demo.json' \
  --warmup 10 \
  --repeat 100
```

Profile the compiler-generated route with Nsight Systems:

```bash
./scripts/perf_profile_nsys.sh perf/runs/nsys_gpu_runner_demo \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=10 \
    --repeat=100 \
    --cubin-format=fatbin
```

Profile the compiler-generated route with Nsight Compute:

```bash
./scripts/perf_profile_ncu.sh perf/runs/ncu_gpu_runner_demo \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=5 \
    --repeat=20 \
    --cubin-format=fatbin
```

## Result Contract

Each backend produces a JSON file with:

- backend name and command
- scalar result when available
- `timings_ms` and latency summary
- correctness status relative to the first successful numeric backend
- artifact paths such as lowered MLIR

External hand CUDA and third-party commands can start simple by printing one
numeric result to stdout. A dedicated future benchmark binary can later emit
its own richer JSON while keeping this top-level comparison format stable.

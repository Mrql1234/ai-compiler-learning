# GPU A10 Workflow

This document describes the recommended split workflow for the current setup:

- local machine: no GPU, used for compiler development
- cloud server: NVIDIA A10, used for real GPU execution and benchmarking

## 1. Recommended split

### Local machine

Use the local machine for:

- `mini` dialect development
- MLIR pass development
- CPU path validation
- GPU IR generation and inspection
- `gpu.launch` / `gpu.module` level debugging

Typical local goal:

- confirm that the compiler can lower
  - `mini.*`
  - to `linalg/tensor/arith`
  - then to `memref`
  - then to `gpu.launch` / `gpu.module`

### Cloud A10 server

Use the cloud server for:

- CUDA runtime execution
- NVVM / PTX generation
- kernel launch validation
- Triton experiments
- correctness checks against CPU
- latency / throughput benchmark

## 2. Current project stages

### Stage A: local compile-only GPU validation

The current project now supports:

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
```

This produces a GPU-oriented IR containing:

- `gpu.launch_func`
- `gpu.module`
- outlined `gpu.func` kernels

This stage does **not** require a local GPU.

### Stage B: cloud A10 backend lowering

On the cloud A10 machine, the next target is:

- attach NVVM target
- lower GPU dialect to NVVM
- lower host side to LLVM
- optionally package GPU binary

Suggested direction:

```text
mini
-> mini-gpu-lowering
-> nvvm-attach-target
-> gpu-lower-to-nvvm-pipeline
```

Recommended command entrypoints:

```bash
./scripts/a10_preflight.sh
./scripts/a10_lower_to_nvvm.sh test/gpu_prep.mlir
```

For NVIDIA A10, the target chip is typically in the Ampere family. In practice,
the exact `sm_*` target should be confirmed on the cloud server before fixing
the final command.

The current script defaults to:

- `GPU_CHIP=sm_86`
- `CUBIN_FORMAT=isa`

This is convenient for inspection. For more realistic cloud execution, switch to:

```bash
CUBIN_FORMAT=fatbin ./scripts/a10_lower_to_nvvm.sh test/gpu_prep.mlir
```

## 3. Suggested near-term milestones

### Milestone 1: local GPU IR

Goal:

- keep `mini-gpu-lowering` stable
- inspect generated `gpu.launch` structure
- verify that elementwise and linear-style examples lower cleanly

### Milestone 2: cloud NVVM path

Goal:

- add a cloud-side command or script that takes GPU IR to NVVM/host LLVM
- verify that a minimal kernel launch works on A10

### Milestone 3: benchmarkable GPU path

Goal:

- compare CPU reference backend vs cloud GPU backend
- add correctness and benchmark scripts

### Milestone 4: Triton route

Goal:

- use Triton for hotspot kernels or fused kernels
- compare Triton backend with the generic MLIR GPU route

### Milestone 5: strategy selection

Goal:

- decide which ops stay on the generic MLIR GPU route
- decide which ops should move to Triton
- decide which ops should become library-backed implementations

The first version can be rule-based. It does not need a full cost model.

## 4. Recommended operator order

Start with:

- elementwise relu
- elementwise add
- elementwise mul
- simple linear / matmul-derived example

Then add:

- fused linear + relu

Then later:

- softmax
- layernorm
- attention-style blocks

## 5. Strategy selection guidance

For the current project, the recommended first split is:

- generic MLIR GPU route
  - `relu`
  - `add`
  - `mul`
- Triton candidate route
  - fused elementwise kernels
  - fused linear epilogues
- library-backed route
  - large `matmul`
  - large `linear`

The important point is to make this layer explicit in the architecture, even if
the first version uses simple rules.

## 6. Immediate next command

Local machine:

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
```

This is the current best checkpoint for “no local GPU, but GPU compiler path is
moving forward”.

Cloud A10 machine:

```bash
./scripts/a10_preflight.sh
CUBIN_FORMAT=fatbin ./scripts/a10_lower_to_nvvm.sh test/gpu_prep.mlir
```

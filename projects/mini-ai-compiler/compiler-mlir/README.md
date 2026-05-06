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
- parse and verify `mini.constant`, `mini.linear`, `mini.relu`, `mini.fused_linear_relu`
- run mini canonicalization / fusion / constant-fold passes
- lower `mini.*` ops to `linalg` / `arith` / `tensor`
- continue into an experimental bufferized CPU-oriented path with standard MLIR passes

## Configure

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

## Design Notes

- Multi-backend lowering roadmap:
  - `LOWERING_ROADMAP.md`
- Local dev + cloud A10 workflow:
  - `GPU_A10_WORKFLOW.md`

## Useful Commands

Lower mini ops to standard tensor/linalg dialects:

```bash
./build/bin/mini-compiler-opt --mini-lower-to-linalg test/lower_to_linalg.mlir
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
```

Run a lowered MLIR module through the local CPU JIT runner:

```bash
./build/bin/mini-compiler-runner test/cpu_runner_demo.mlir --entry-point-result=f32
```

Prepare `mini.*` programs for a later GPU/Triton route:

```bash
./build/bin/mini-compiler-opt --mini-gpu-prep test/gpu_prep.mlir
```

Lower further into GPU launch/module form without needing a local GPU:

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
```

Preflight the cloud A10 NVVM toolchain:

```bash
./scripts/a10_preflight.sh
```

Run the staged A10 NVVM lowering pipeline:

```bash
./scripts/a10_lower_to_nvvm.sh test/gpu_prep.mlir
```

Translate the LLVM dialect output into textual LLVM IR:

```bash
./build/bin/mini-compiler-opt --mini-cpu-lowering test/cpu_pipeline.mlir \
  | /path/to/matching/mlir-translate --mlir-to-llvmir
```

Important:

- `mlir-translate` should come from the **same LLVM/MLIR build** as the one used to build `mini-compiler-opt`
- mixing the custom source build with older system binaries can fail on newer LLVM dialect ops such as `llvm.mlir.poison`

## Expected Next Steps

1. extend the runnable CPU path beyond the current demo entry
2. add more mini ops and MLIR-native optimization passes
3. connect the Python bridge to emit stable MLIR input for this toolchain
4. push `mini-gpu-lowering` toward cloud-side NVVM execution on A10
5. map the same high-level ops into a Triton-oriented backend path

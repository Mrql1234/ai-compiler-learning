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
./build/bin/mini-compiler-opt \
  --mini-lower-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --drop-equivalent-buffer-results \
  --buffer-results-to-out-params \
  --convert-bufferization-to-memref \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --convert-index-to-llvm \
  --expand-realloc \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  test/lower_to_llvm.mlir
```

## Expected Next Steps

1. add execution support on top of the LLVM-dialect CPU path
2. wrap the standard CPU lowering pipeline behind a project-local driver/preset
3. add more mini ops and MLIR-native optimization passes
4. connect the Python bridge to emit stable MLIR input for this toolchain

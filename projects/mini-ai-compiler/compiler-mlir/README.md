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

The current version is a bootstrap out-of-tree MLIR project skeleton.

It is intended to:

- compile against an existing LLVM/MLIR build
- parse `mini.*` operations through a registered dialect skeleton
- provide pass registration hooks
- provide a smoke-test tool entrypoint

## Configure

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

## Expected Next Steps

1. define bridge-text parsing path
2. materialize `mini` ops beyond unknown-op skeleton behavior
3. implement MLIR-native passes
4. lower to LLVM and Triton/GPU paths

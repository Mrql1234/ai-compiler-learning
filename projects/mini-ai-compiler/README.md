# Mini AI Compiler

`Mini AI Compiler` is now a dual-track project:

- A **Python prototype track** for frontend import, bridge export, reference execution, validation, and benchmark.
- A **C++ MLIR track** under `compiler-mlir/` for the real compiler pipeline: dialect, passes, lowering, and backend integration.

The intended main pipeline is:

`PyTorch FX / ONNX -> Python bridge -> MLIR module/dialect -> MLIR passes -> CPU(LLVM) + Triton/GPU -> validation`

## Python Track

- FX importer
- ONNX importer MVP
- Prototype IR
- Prototype passes
- CPU reference backend
- Triton/GPU strategy selection and lowering plan
- Benchmark and dump tools

## MLIR Track

- Out-of-tree C++ MLIR subproject
- Dialect skeleton
- Pass registration skeleton
- Compiler driver skeleton
- Smoke test layout

## Dependencies

```bash
python3 -m pip install --user -r requirements-dev.txt
```

## Layout

```text
projects/mini-ai-compiler/
  frontend/
  ir/
  passes/
  backend/cpu/
  tools/
  tests/
  benchmarks/
  compiler-mlir/
```

## Quick Start

```bash
cd projects/mini-ai-compiler
python -m tools.run_mlp_example
python -m tools.dump_ir
python -m tools.export_mlir
python -m tools.export_bridge_mlir
python -m tools.run_triton_example
python -m tools.run_mlir_canonicalize_demo
python -m benchmarks.bench_mlp
python -m unittest discover -s tests
```

## MLIR Subproject

The `compiler-mlir/` directory is the formal compiler track.

Typical configure flow:

```bash
cd projects/mini-ai-compiler/compiler-mlir
cmake -S . -B build \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

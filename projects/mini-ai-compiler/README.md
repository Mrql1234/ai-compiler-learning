# Mini AI Compiler

An educational end-to-end AI compiler project that imports small PyTorch FX graphs, lowers them into a custom IR, applies graph optimizations, and executes them on a CPU reference backend.

## Phase 1 Scope

- PyTorch FX importer
- Custom IR
- Constant fold
- DCE
- CPU backend
- MLP example

## Phase 2 Scope

- `linear + relu` fusion
- IR dump files under `artifacts/`
- MLP benchmark
- ONNX importer MVP for a small op subset

## Phase 3 Scope

- Triton lowering plan
- Guarded Triton executor entrypoint
- `run_triton_example` demo for CUDA-capable environments

## Phase 4 Scope

- MLIR-style textual IR export
- `.mlir` artifact output
- MLIR-facing bridge from custom IR concepts
- MLIR-style pattern/rewrite prototype

## Dependencies

```bash
python3 -m pip install --user -r requirements-dev.txt
```

## Quick Start

```bash
cd projects/mini-ai-compiler
python -m tools.run_mlp_example
python -m tools.dump_ir
python -m tools.export_mlir
python -m tools.run_triton_example
python -m tools.run_mlir_canonicalize_demo
python -m benchmarks.bench_mlp
python -m unittest discover -s tests
```

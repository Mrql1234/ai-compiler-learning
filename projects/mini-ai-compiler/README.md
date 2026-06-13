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

## Triton 算子 Agent 原型

`compiler-mlir/perf` 现在新增了一套“面向 Triton 的算子开发与自动迭代优化 Agent 原型”，目标是把下面这条闭环固化成统一入口：

`结构化算子规格 -> 候选配置生成 -> 正确性/benchmark -> profiling 诊断 -> 下一轮优化建议`

当前状态：

- `fused_linear_relu`
  - 已接入可执行的 Triton benchmark / profile 工作流
  - 可用 `plan` / `tune` / `analyze` 三种模式
- `matmul`、`softmax`、`layernorm`
  - 已接入统一规格解析、候选配置生成、经验记忆和 Nsight 诊断接口
  - 当前仍属于 `planner-only` 原型，后续可继续补独立 benchmark

主要入口文件：

- `projects/mini-ai-compiler/compiler-mlir/scripts/triton_operator_agent.py`
- `projects/mini-ai-compiler/compiler-mlir/scripts/triton_operator_agent_lib.py`
- `projects/mini-ai-compiler/compiler-mlir/perf/specs/`

常用命令：

```bash
cd projects/mini-ai-compiler/compiler-mlir

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

如果已经有 `ncu --page details` 导出的文本，也可以单独走诊断模式：

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode analyze \
  --ncu-details /path/to/iter_best_ncu_details.txt \
  --run-dir perf/runs/agent_runs/analyze_linear_relu
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

Weight-only INT8 quantization entry points currently live in:

- `projects/mini-ai-compiler/compiler-mlir/test/quantize_weights.mlir`
- `projects/mini-ai-compiler/compiler-mlir/test/quantized_lower_to_linalg.mlir`
- `projects/mini-ai-compiler/compiler-mlir/test/quantized_gpu_lowering.mlir`
- `projects/mini-ai-compiler/compiler-mlir/test/quantized_qlinear_late_stage.mlir`

Typical commands:

```bash
cd projects/mini-ai-compiler/compiler-mlir
./build/bin/mini-compiler-opt --mini-quantize-weights test/quantize_weights.mlir
./build/bin/mini-compiler-opt --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion,mini-quantize-weights,mini-lower-to-linalg))' test/quantized_qlinear_late_stage.mlir
./build/bin/mini-compiler-opt --mini-quantized-gpu-prep test/quantized_lower_to_linalg.mlir
./build/bin/mini-compiler-opt --mini-quantized-gpu-lowering test/quantized_gpu_lowering.mlir
./build/bin/mini-compiler-opt --mini-quantized-cpu-lowering test/quantized_lower_to_linalg.mlir
./build/bin/mini-compiler-gpu-runner --quantized test/quantized_runner_demo.mlir
```

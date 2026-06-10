# Mini AI Compiler

`Mini AI Compiler` 当前是一个围绕 `AI 编译器 / MLIR / Triton` 学习与实验的双轨项目：

- `Python` 原型轨：负责前端导入、教学型 IR、参考执行、验证与 benchmark。
- `C++ MLIR` 主线轨：位于 `compiler-mlir/`，负责正式的 dialect、pass、lowering 与后端集成。

当前主链路可以概括为：

`PyTorch FX / ONNX -> Python bridge -> MLIR module/dialect -> MLIR passes -> CPU(LLVM) + Triton/GPU -> validation`

## 当前能力

### Python 原型轨

- `FX importer`
- `ONNX importer` MVP
- 自定义图 IR
- Python pass 原型
- CPU reference backend
- Triton/GPU strategy selection 与 lowering plan
- benchmark 与 dump 工具

### MLIR 主线轨

- out-of-tree `C++ MLIR` 子工程
- dialect skeleton
- pass registration skeleton
- compiler driver skeleton
- smoke test 布局

## 依赖

```bash
python3 -m pip install --user -r requirements-dev.txt
```

## 目录结构

```text
projects/mini-ai-compiler/
  frontend/
  ir/
  passes/
  backend/cpu/
  backend/triton/
  tools/
  tests/
  benchmarks/
  compiler-mlir/
```

## 快速开始

```bash
cd projects/mini-ai-compiler
python3 -m tools.run_mlp_example
python3 -m tools.dump_ir
python3 -m tools.export_mlir
python3 -m tools.export_bridge_mlir
python3 -m tools.run_triton_example
python3 -m tools.run_mlir_canonicalize_demo
python3 -m benchmarks.bench_mlp
python3 -m unittest discover -s tests
```

## Triton 算子作品集文档

为了面向芯片厂 `AI 编译器 / 后端 / kernel compiler` 岗位准备一个“小而强”的作品集，当前已经新增文档：

- [TRITON_OPERATOR_PORTFOLIO.md](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/TRITON_OPERATOR_PORTFOLIO.md)

这份文档聚焦 5 个核心算子：

- `triton_matmul.py`
- `triton_fused_linear_relu.py`
- `triton_softmax.py`
- `triton_layernorm.py`
- `triton_flash_attention_simplified.py`

文档中已经给出：

- 学习目标与岗位定位
- 推荐目录布局
- 每个算子的公式、shape、Triton 实现思路
- 正确性验证方式
- benchmark 方式
- 与编译器优化的关系
- 分阶段实施计划

### 作品集建议入口文件

下面这些是文档中约定的建议入口文件，后续代码实现时建议保持不变：

- `portfolio/triton_ops/triton_matmul.py`
- `portfolio/triton_ops/triton_fused_linear_relu.py`
- `portfolio/triton_ops/triton_softmax.py`
- `portfolio/triton_ops/triton_layernorm.py`
- `portfolio/triton_ops/triton_flash_attention_simplified.py`

### 作品集建议运行命令

下面这些是文档中约定的建议运行命令，后续代码落地后可以直接按这个模块路径执行：

```bash
cd projects/mini-ai-compiler
python3 -m portfolio.triton_ops.triton_matmul
python3 -m portfolio.triton_ops.triton_fused_linear_relu
python3 -m portfolio.triton_ops.triton_softmax
python3 -m portfolio.triton_ops.triton_layernorm
python3 -m portfolio.triton_ops.triton_flash_attention_simplified
```

建议同步配套以下验证入口：

```bash
cd projects/mini-ai-compiler
python3 -m portfolio.tests.test_triton_matmul
python3 -m portfolio.tests.test_triton_fused_linear_relu
python3 -m portfolio.tests.test_triton_softmax
python3 -m portfolio.tests.test_triton_layernorm
python3 -m portfolio.tests.test_triton_flash_attention_simplified
python3 -m portfolio.benchmarks.bench_triton_matmul
python3 -m portfolio.benchmarks.bench_triton_fused_linear_relu
python3 -m portfolio.benchmarks.bench_triton_softmax
python3 -m portfolio.benchmarks.bench_triton_layernorm
python3 -m portfolio.benchmarks.bench_triton_flash_attention_simplified
```

## MLIR 子工程

`compiler-mlir/` 是正式的 MLIR 编译器主线。

典型配置流程：

```bash
cd projects/mini-ai-compiler/compiler-mlir
cmake -S . -B build \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build build
```

当前权重量化相关入口文件包括：

- `compiler-mlir/test/quantize_weights.mlir`
- `compiler-mlir/test/quantized_lower_to_linalg.mlir`
- `compiler-mlir/test/quantized_gpu_lowering.mlir`
- `compiler-mlir/test/quantized_qlinear_late_stage.mlir`

典型命令：

```bash
cd projects/mini-ai-compiler/compiler-mlir
./build/bin/mini-compiler-opt --mini-quantize-weights test/quantize_weights.mlir
./build/bin/mini-compiler-opt --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion,mini-quantize-weights,mini-lower-to-linalg))' test/quantized_qlinear_late_stage.mlir
./build/bin/mini-compiler-opt --mini-quantized-gpu-prep test/quantized_lower_to_linalg.mlir
./build/bin/mini-compiler-opt --mini-quantized-gpu-lowering test/quantized_gpu_lowering.mlir
./build/bin/mini-compiler-opt --mini-quantized-cpu-lowering test/quantized_lower_to_linalg.mlir
./build/bin/mini-compiler-gpu-runner --quantized test/quantized_runner_demo.mlir
```

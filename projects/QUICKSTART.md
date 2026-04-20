# 快速启动指南 🚀

> 5 分钟开始第一个项目

---

## 前置检查

```bash
# 检查 Python
python --version  # 需要 3.8+

# 检查 GPU
nvidia-smi  # 需要 NVIDIA GPU + CUDA 11.7+

# 检查 Git
git --version
```

---

## 项目 1: Triton 算子库（推荐首选）

### 1. 创建虚拟环境

```bash
cd /home/admin/.openclaw/workspace/projects/triton-kernel-library

python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或：venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; import triton; print('✓ PyTorch:', torch.__version__); print('✓ Triton:', triton.__version__); print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### 4. 运行第一个测试

```bash
# 测试 LayerNorm
python kernels/layernorm.py
```

**预期输出**：
```
============================================================
Triton LayerNorm 测试 (batch=32, seq=512, hidden=768)
============================================================

1. 正确性测试...
   最大差异：0.000012
   结果正确：✓

2. 性能测试...
   Triton:   0.245 ms
   PyTorch:  0.312 ms
   加速比：  1.27x

✓ 测试完成！
```

### 5. 运行完整 Benchmark

```bash
cd benchmarks
python benchmark_all.py
```

### 6. 运行正确性测试

```bash
cd tests
python test_correctness.py
```

---

## 项目 2: TVM 模型部署

### 1. 检查 TVM

```bash
cd /home/admin/.openclaw/workspace/projects/tvm-model-deployment

python -c "import tvm; print('TVM:', tvm.__version__)"
```

如果未安装，选择以下一种方式：

**方式 A：pip 安装（推荐）**
```bash
pip install --upgrade apache-tvm
```

**方式 B：源码编译（高级）**
```bash
# 参考：https://tvm.apache.org/docs/install/from_source.html
```

### 2. 运行矩阵乘法优化

```bash
cd schedule
python matmul_opt.py
```

**预期输出**：
```
======================================================================
矩阵乘法调度对比 (M=512, N=512, K=512)
======================================================================
朴素版本                   125.432 ms  ( 1.00x)
分块优化                    62.145 ms  ( 2.02x)
分块 + 并行 + 向量          14.234 ms  ( 8.81x)
完整优化                    13.892 ms  ( 9.03x)
```

---

## 项目 3: MLIR Pass 开发

### 1. 编译 LLVM（一次性，约 1 小时）

```bash
# 克隆
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/15.x

# 创建构建目录
mkdir build && cd build

# 配置
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release

# 编译
ninja -j$(nproc)
```

### 2. 构建项目

```bash
cd /home/admin/.openclaw/workspace/projects/mlir-passes
mkdir build && cd build

cmake .. -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
         -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir

ninja -j$(nproc)
```

### 3. 测试 Pass

```bash
# 测试常量折叠
./bin/mlir-opt --constant-fold ../test/constant_fold.mlir
```

---

## 📊 进度追踪

创建进度文件：

```bash
cd /home/admin/.openclaw/workspace/projects
touch progress.md
```

记录每天进展：

```markdown
# 学习进度

## 2026-04-20
- [x] 创建项目结构
- [x] 安装 Triton 环境
- [x] 跑通 LayerNorm 测试
- [ ] 实现 GELU

## 2026-04-21
- [ ] ...
```

---

## 🆘 遇到问题？

### Triton 安装失败

```bash
# 检查 CUDA 版本
nvcc --version

# Triton 需要 CUDA 11.7+
# 如果不匹配，升级 CUDA 或使用 Docker
```

### TVM 导入错误

```bash
# 检查 Python 版本
python --version  # 需要 3.8+

# 重新安装
pip uninstall apache-tvm
pip install apache-tvm
```

### MLIR 编译错误

```bash
# 检查 CMake 版本
cmake --version  # 需要 3.20+

# 清理重新编译
rm -rf build
mkdir build && cd build
cmake ...
```

---

## 📞 获取帮助

- **GitHub Issues**: 在项目仓库提 issue
- **Discord**: AI 编译器学习社区
- **文档**: 每个项目的 README.md

---

## ✅ 检查清单

开始每个项目前，确保：

- [ ] 环境安装成功
- [ ] 示例代码能运行
- [ ] 理解基本概念
- [ ] 准备好笔记工具

---

_最后更新：2026-04-20 | 有问题？查看各项目的 README.md_

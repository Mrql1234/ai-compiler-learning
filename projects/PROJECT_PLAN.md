# 三项目实战计划 📅

> MLIR + Triton + TVM 三个月学习路线  
> 适合：有 GPU、想深入 AI 编译器开发的开发者

---

## 📊 项目总览

| 项目 | 技术栈 | 周期 | 难度 | 产出 |
|------|--------|------|------|------|
| **Triton 算子库** | Triton, PyTorch, CUDA | 3-4 周 | ⭐⭐⭐ | 5 个算子 + Benchmark |
| **TVM 模型部署** | TVM, LLVM, Python | 4-5 周 | ⭐⭐⭐⭐ | 3 个模型 + 调度优化 |
| **MLIR Pass 开发** | C++, MLIR, LLVM | 5-6 周 | ⭐⭐⭐⭐⭐ | 3 个 Pass + 开源贡献 |

---

## 🗓️ 详细时间规划

### 第 1 个月：Triton 算子库

**目标**：掌握 GPU 算子开发，理解内存层次和并行模型

#### 第 1 周：环境 + 基础算子
- [ ] 安装 Triton + PyTorch
- [ ] 跑通向量加法示例
- [ ] 实现 LayerNorm
- [ ] 正确性测试（对比 PyTorch）

#### 第 2 周：激活函数 + 归一化
- [ ] 实现 GELU
- [ ] 实现 RMSNorm
- [ ] 性能 Benchmark
- [ ] 写技术博客（可选）

#### 第 3 周：位置编码 + Attention
- [ ] 实现 RoPE
- [ ] 实现 FlashAttention（简化版）
- [ ] Causal mask 支持
- [ ] 显存占用测试

#### 第 4 周：优化 + 文档
- [ ] BLOCK_SIZE 调优
- [ ] 完整 Benchmark 报告
- [ ] README 完善
- [ ] GitHub 发布

**预期产出**：
- GitHub 仓库：`triton-kernel-library`
- 5 个 Triton 算子
- 性能数据：1.4-3.6x 加速

---

### 第 2 个月：TVM 模型部署

**目标**：掌握编译器端到端流程，理解调度优化

#### 第 1 周：TVM 入门
- [ ] 编译 TVM（或安装预编译版）
- [ ] 跑通 TE/TIR 示例
- [ ] 理解 Schedule 概念
- [ ] 实现矩阵乘法调度

#### 第 2 周：调度优化
- [ ] 分块 (Tiling) 优化
- [ ] 并行化 (Parallel)
- [ ] 向量化 (Vectorize)
- [ ] 性能对比实验

#### 第 3 周：模型编译
- [ ] ResNet-18 编译
- [ ] MobileNetV2 编译
- [ ] 端到端推理测试
- [ ] 正确性验证

#### 第 4-5 周：自动调优 + 文档
- [ ] 轻量参数搜索
- [ ] 调优结果记录
- [ ] Benchmark 报告
- [ ] 技术文档

**预期产出**：
- GitHub 仓库：`tvm-model-deployment`
- 3 个模型编译流程
- 性能数据：2-5x 加速
- 调度优化笔记

---

### 第 3 个月：MLIR Pass 开发

**目标**：深入编译器核心，理解 IR 和 Pass 架构

#### 第 1 周：LLVM/MLIR 环境
- [ ] 编译 LLVM（含 MLIR）
- [ ] 理解 MLIR 基础概念
- [ ] 跑通 Toy 教程
- [ ] 搭建项目框架

#### 第 2 周：常量折叠 Pass
- [ ] 实现 AddI/MulI 折叠
- [ ] 实现 AddF 折叠
- [ ] 编写测试用例
- [ ] 性能测试

#### 第 3 周：死代码消除 Pass
- [ ] 理解 SSA 和用途链
- [ ] 实现 DCE Pass
- [ ] 链式消除优化
- [ ] 测试验证

#### 第 4 周：算子融合 Pass
- [ ] 设计融合模式
- [ ] 实现 Fusion Pass
- [ ] 性能收益分析
- [ ] 文档完善

#### 第 5-6 周：开源贡献
- [ ] 代码整理
- [ ] 阅读 LLVM 贡献指南
- [ ] 准备 PR
- [ ] 提交（或模拟 PR）

**预期产出**：
- GitHub 仓库：`mlir-passes`
- 3 个 MLIR Pass
- LLVM PR（或技术博客）
- 编译器架构理解

---

## 📈 实验方法

### 1. 正确性验证

所有项目都必须有正确性测试：

```python
# Triton/TVM
output_triton = kernel(input)
output_ref = pytorch_reference(input)
diff = torch.max(torch.abs(output_triton - output_ref))
assert diff < 1e-4, "数值差异过大"
```

```cpp
// MLIR
// RUN: mlir-opt --constant-fold %s | FileCheck %s
```

### 2. 性能测试

使用精确计时：

```python
# GPU 用 Event
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... run kernel ...
end.record()
torch.cuda.synchronize()
time_ms = start.elapsed_time(end)
```

```bash
# CPU 用 time
time python benchmark.py
```

### 3. 数据记录

用 JSON 记录所有实验数据：

```json
{
  "kernel": "layernorm",
  "shape": [32, 512, 768],
  "triton_ms": 1.8,
  "pytorch_ms": 2.5,
  "speedup": 1.39,
  "gpu": "A100",
  "timestamp": "2026-04-20T18:00:00"
}
```

### 4. 可视化

用 matplotlib 画对比图：

```python
import matplotlib.pyplot as plt

kernels = ['LayerNorm', 'GELU', 'RMSNorm', 'FlashAttn']
speedups = [1.39, 1.50, 1.40, 3.62]

plt.bar(kernels, speedups)
plt.ylabel('Speedup (x)')
plt.title('Triton vs PyTorch Performance')
plt.savefig('benchmark.png')
```

---

## 🎯 成功标准

### Triton 项目
- [ ] 5 个算子全部通过正确性测试
- [ ] 性能超越 PyTorch（平均 1.5x+）
- [ ] GitHub 仓库有完整 README
- [ ] Benchmark 脚本可复现

### TVM 项目
- [ ] 3 个模型成功编译
- [ ] 调度优化带来 2x+ 加速
- [ ] 理解 TVM 编译流程
- [ ] 技术文档完整

### MLIR 项目
- [ ] 3 个 Pass 通过测试
- [ ] 理解 Dialect/Operation/Pass
- [ ] 能读懂 MLIR IR
- [ ] 尝试提交 LLVM PR

---

## 💡 常见问题

### Q1: 三个项目都要做吗？

**A**: 建议至少完成前两个（Triton + TVM）。MLIR 难度较高，可根据时间决定。

### Q2: 没有 GPU 怎么办？

**A**: 
- Triton：无法运行（必须 NVIDIA GPU）
- TVM：可用 CPU（LLVM 后端）
- MLIR：可用 CPU

### Q3: 每个项目要花多少时间？

**A**: 
- Triton: 50-80 小时
- TVM: 60-100 小时
- MLIR: 80-120 小时

### Q4: 如何平衡工作和学习？

**A**: 
- 每天 2 小时，周末 6-8 小时
- 先完成最小可用版本（MVP）
- 逐步迭代优化

### Q5: 项目完成后有什么用？

**A**:
- **简历亮点**：AI 编译器开发经验
- **面试加分**：深入理解底层优化
- **实际技能**：GPU 编程、编译器优化
- **开源贡献**：LLVM/MLIR PR

---

## 📚 学习资源

### Triton
- [官方文档](https://triton-lang.org/)
- [Triton Tutorial](https://github.com/wookayin/triton-tutorial)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)

### TVM
- [官方教程](https://tvm.apache.org/docs/)
- [自动调优指南](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/)
- [TVM 源码](https://github.com/apache/tvm)

### MLIR
- [MLIR 教程](https://mlir.llvm.org/docs/Tutorials/)
- [Toy 示例](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [LLVM 贡献指南](https://llvm.org/docs/Contributing.html)

---

## 🚀 立即开始

### 第一步（今天）
1. Fork 三个项目仓库
2. 安装 Triton 环境
3. 跑通第一个示例（向量加法）

### 第二步（本周）
1. 实现 LayerNorm
2. 写测试脚本
3. 记录性能数据

### 第三步（本月）
1. 完成 Triton 项目
2. 开始 TVM 项目
3. 写技术博客

---

_计划创建：2026-04-20 | 适合人群：有 Python 基础、想深入 AI 编译器 | 预计周期：3 个月_

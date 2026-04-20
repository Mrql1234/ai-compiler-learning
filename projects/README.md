# AI 编译器实战项目合集 🚀

> MLIR + Triton + TVM 三个完整项目，助你深入 AI 编译器开发

---

## 📁 项目导航

| 项目 | 技术栈 | 难度 | 周期 | 状态 |
|------|--------|------|------|------|
| [**Triton 算子库**](./triton-kernel-library/) | Triton, PyTorch, CUDA | ⭐⭐⭐ | 3-4 周 | ✅ 就绪 |
| [**TVM 模型部署**](./tvm-model-deployment/) | TVM, LLVM, Python | ⭐⭐⭐⭐ | 4-5 周 | ✅ 就绪 |
| [**MLIR Pass 开发**](./mlir-passes/) | C++, MLIR, LLVM | ⭐⭐⭐⭐⭐ | 5-6 周 | ✅ 就绪 |

---

## 🎯 为什么做这些项目？

### 1. 技能稀缺性

AI 编译器开发是**高度稀缺**的技能：
- 市场上大多数是应用层开发
- 底层优化人才供不应求
- 薪资溢价明显

### 2. 技术深度

这三个项目覆盖 AI 编译器核心领域：
- **Triton**：GPU 算子开发（最接近硬件）
- **TVM**：端到端编译流程（承上启下）
- **MLIR**：编译器架构（最底层）

### 3. 简历亮点

每个项目都能写进简历：
```
 Triton 算子库
 - 实现 5 个 LLM 核心算子（LayerNorm, GELU, FlashAttention 等）
 - 性能超越 PyTorch 原生 1.4-3.6 倍
 - GitHub: github.com/yourname/triton-kernel-library

 TVM 模型部署
 - 编译 ResNet/MobileNet 到 CPU，2-5 倍加速
 - 实现分块/并行/向量化调度优化
 - 理解自动调优流程

 MLIR Pass 开发
 - 开发常量折叠/死代码消除 Pass
 - 深入理解 Dialect/Operation/Pass 架构
 - 向 LLVM 开源项目提交 PR（或准备中）
```

---

## 🚀 快速开始

### 5 分钟上手

```bash
# 1. 选择第一个项目（推荐 Triton）
cd triton-kernel-library

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试
python kernels/layernorm.py
```

**详细指南**：[QUICKSTART.md](./QUICKSTART.md)

---

## 📊 预期成果

### Triton 项目
```
性能对比（A100）：
┌─────────────┬───────────┬────────────┬─────────┐
│ 算子        │ PyTorch   │ Triton     │ 加速比  │
├─────────────┼───────────┼────────────┼─────────┤
│ LayerNorm   │ 2.5 ms    │ 1.8 ms     │ 1.39x   │
│ GELU        │ 1.8 ms    │ 1.2 ms     │ 1.50x   │
│ RMSNorm     │ 2.1 ms    │ 1.5 ms     │ 1.40x   │
│ RoPE        │ 3.2 ms    │ 2.1 ms     │ 1.52x   │
│ FlashAttn   │ 45.2 ms   │ 12.5 ms    │ 3.62x   │
└─────────────┴───────────┴────────────┴─────────┘
```

### TVM 项目
```
模型编译加速（CPU）：
┌─────────────┬───────────┬────────────┬─────────┐
│ 模型        │ PyTorch   │ TVM 优化   │ 加速比  │
├─────────────┼───────────┼────────────┼─────────┤
│ ResNet-18   │ 85 ms     │ 42 ms      │ 2.02x   │
│ MobileNetV2 │ 45 ms     │ 18 ms      │ 2.50x   │
│ BERT-Base   │ 120 ms    │ 65 ms      │ 1.85x   │
└─────────────┴───────────┴────────────┴─────────┘
```

### MLIR 项目
```
Pass 优化效果：
┌─────────────┬───────────┬────────────┬─────────┐
│ Pass        │ 优化前    │ 优化后     │ 代码减少│
├─────────────┼───────────┼────────────┼─────────┤
│ 常量折叠    │ 15 ops    │ 6 ops      │ 60%     │
│ 死代码消除  │ 20 ops    │ 12 ops     │ 40%     │
│ 算子融合    │ 25 ops    │ 15 ops     │ 40%     │
└─────────────┴───────────┴────────────┴─────────┘
```

---

## 📅 学习路线

```
第 1 个月                    第 2 个月                    第 3 个月
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Triton 算子库  │        │  TVM 模型部署   │        │  MLIR Pass 开发  │
│                 │        │                 │        │                 │
│  Week 1: 基础   │        │  Week 1: TVM    │        │  Week 1: LLVM   │
│  Week 2: 算子   │        │  Week 2: 调度   │        │  Week 2: 常量   │
│  Week 3: Attn   │        │  Week 3: 模型   │        │  Week 3: DCE    │
│  Week 4: 优化   │        │  Week 4: 调优   │        │  Week 4: 融合   │
│                 │        │                 │        │  Week 5: PR     │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

**详细计划**：[PROJECT_PLAN.md](./PROJECT_PLAN.md)

---

## 🛠️ 环境要求

### 硬件
- **GPU**: NVIDIA RTX 3090+/A100（Triton 必需）
- **CPU**: 多核（TVM 优化需要）
- **内存**: 16GB+（LLVM 编译需要）

### 软件
- Python 3.8+
- CUDA 11.7+
- CMake 3.20+（MLIR 需要）
- LLVM 15+（MLIR 需要）

---

## 📚 前置知识

### 必需
- Python 编程
- 深度学习基础（PyTorch）
- 线性代数基础

### 加分（非必需）
- CUDA 编程基础
- 编译器概念
- C++ 基础（MLIR 需要）

**零基础？** 先看你的学习笔记：
- [笔记 02: MLIR 深入](../ai-compiler-learning/notes/02-mlir-deep-dive.md)
- [笔记 07: Triton 编程](../ai-compiler-learning/notes/07-triton-programming.md)
- [笔记 04: TVM 调度](../ai-compiler-learning/notes/04-tvm-scheduling-optimization.md)

---

## 🎓 学习建议

### 1. 先跑通，再理解

不要一开始就追求完美理解所有细节：
```bash
# 先让代码跑起来
python kernels/layernorm.py

# 看到性能数据后，再深入理解为什么
```

### 2. 记录实验数据

用表格记录每次实验：
```markdown
| 日期 | 算子 | 形状 | Triton | PyTorch | 加速比 | 备注 |
|------|------|------|--------|---------|--------|------|
| 4/20 | LN   | 32x512x768 | 1.8ms | 2.5ms | 1.39x | BLOCK_SIZE=1024 |
```

### 3. 写技术博客

教是最好的学：
- 记录实现细节
- 分享性能数据
- 发布到知乎/掘金

### 4. 参与开源

- 给项目提 Issue
- 修复小 bug
- 提交 PR

---

## 📞 支持

### 遇到问题？

1. **查文档**：每个项目的 README.md
2. **查笔记**：你的学习笔记
3. **提 Issue**：GitHub Issues
4. **问社区**：AI 编译器学习群

### 学习社区

- [Triton GitHub](https://github.com/openai/triton)
- [TVM GitHub](https://github.com/apache/tvm)
- [MLIR Discourse](https://discourse.llvm.org/)

---

## 🏆 完成奖励

完成所有项目后，你将获得：

✅ **硬技能**
- GPU 算子开发能力
- 编译器优化理解
- 性能分析技能

✅ **软技能**
- 复杂项目管理
- 技术文档写作
- 开源协作经验

✅ **简历加成**
- 3 个完整项目
- 性能数据支撑
- GitHub 仓库链接

✅ **面试优势**
- 深入底层理解
- 实际问题解决
- 技术热情证明

---

## 📝 License

MIT License - 可自由使用和修改

---

_项目创建：2026-04-20 | 作者：ql | 指导：cx330 ✨_

**开始你的 AI 编译器之旅吧！** 🚀

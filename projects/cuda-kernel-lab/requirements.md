# CUDA Kernel Lab 需求说明

## 目标

建立一个面向学习和实习展示的手写 CUDA 项目，覆盖常见算子的：

- kernel 实现
- 正确性验证
- benchmark
- Nsight 性能分析

## 当前阶段需求

第一阶段先满足下面几点：

1. 可以独立构建，不和 `triton-kernel-library` 混在一起
2. 有最小可运行的 CUDA 示例
3. 有统一的 build / run / profile 入口
4. 文档中明确入口文件和运行命令
5. 后续可以逐步扩展到更复杂算子

## 非目标

当前不追求：

- 完整 PyTorch Extension 封装
- 自动调参框架
- 生产级 kernel 最优性能
- 一次性覆盖所有 LLM 算子


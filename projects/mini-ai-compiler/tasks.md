# Tasks: Mini AI Compiler

## Phase 1: 最小闭环

- [x] T1. 建立项目目录骨架
  - 创建 `frontend/`、`ir/`、`passes/`、`backend/cpu/`、`runtime/`、`tests/`、`examples/`、`tools/`
  - 创建必要的 `__init__.py`

- [x] T2. 定义最小 IR 数据结构
  - 实现 `Graph`
  - 实现 `Node`
  - 实现 `Value`
  - 实现 `TensorType`
  - 支持打印、遍历、节点替换、节点删除

- [x] T3. 实现 PyTorch FX importer MVP
  - 支持 `placeholder`
  - 支持 `output`
  - 支持 `call_module`
  - 支持 `call_function`
  - 第一版覆盖 `Linear / ReLU / add / mul / matmul`

- [x] T4. 实现 CPU reference backend MVP
  - 支持常量、输入绑定、拓扑执行
  - 支持 `matmul`
  - 支持 `add`
  - 支持 `mul`
  - 支持 `relu`

- [x] T5. 实现 Constant Fold Pass MVP
  - 支持 `Add(Const, Const) -> Const`
  - 支持 `Mul(Const, Const) -> Const`
  - 支持固定点重复运行直到无变化

- [x] T6. 实现 DCE Pass MVP
  - 删除无用户且非图输出节点
  - 保持图输出语义不变

- [x] T7. 提供 IR dump 能力
  - 输出原始 IR
  - 输出优化后 IR
  - 提供简单 CLI 或脚本入口

- [x] T8. 提供 MLP 示例
  - 编写最小 MLP 模型
  - 通过 FX 导入
  - 跑通 CPU backend
  - 对照 PyTorch eager 输出

- [x] T9. 添加基础测试
  - IR 结构测试
  - importer 测试
  - CPU backend 测试
  - constant fold / DCE 测试

## Phase 2: 优化与可视化
- [x] T10. 实现 Fusion Pass
- [x] T11. 增强 IR dump 与 diff 展示
- [x] T12. 添加 benchmark 脚本
- [x] T13. 实现 ONNX importer MVP

## Phase 3: Triton Backend
- [x] T14. 实现 Triton kernel MVP (`matmul/add/relu`)
- [x] T15. 实现 Triton executor
- [x] T16. 支持 fused op lowering
- [x] T17. 添加 CPU / Triton 对照测试与 benchmark

## Phase 4: MLIR 升级版
- [x] T18. 输出 MLIR 风格 IR
- [x] T19. 评估 IR 到 MLIR 概念映射
- [x] T20. 迁移一个 pass 到 MLIR-based 实现

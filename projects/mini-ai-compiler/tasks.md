# Tasks: Mini AI Compiler

## Phase A：文档与架构重置

- [x] A1. 重写 `requirements.md`
- [x] A2. 重写 `design.md`
- [x] A3. 重写 `tasks.md`
- [x] A4. 更新 `README.md`，明确双轨架构

## Phase B：MLIR 工程骨架

- [x] B1. 新增 `compiler-mlir/` 子工程目录
- [x] B2. 接通官方 LLVM/MLIR CMake skeleton
- [x] B3. 提供 dialect 注册骨架
- [x] B4. 提供 pass 注册骨架
- [x] B5. 提供 driver/tool 骨架
- [x] B6. 提供 smoke test 样例

## Phase C：前端桥接

- [x] C1. 明确 bridge 策略优先为文本桥接
- [x] C2. 新增 Python bridge 导出工具
- [ ] C3. 打通 bridge 文本 -> MLIR module 解析
- [ ] C4. 提供 FX / ONNX 样例桥接输入

## Phase D：MLIR Pass

- [ ] D1. 在 MLIR 工程中新建 canonicalize pass 骨架
- [ ] D2. 在 MLIR 工程中新建 constant fold pass 骨架
- [ ] D3. 在 MLIR 工程中新建 DCE pass 骨架
- [ ] D4. 在 MLIR 工程中新建 fusion pass 骨架
- [ ] D5. 添加 `mlir-opt` 级测试

## Phase E：CPU 路线

- [x] E1. 设计 `Mini -> LLVM` lowering 路线
- [x] E2. 跑通 `MLIR -> LLVM IR -> CPU`
- [x] E3. 用 `MLP` 做 CPU 正式链路验证
- [ ] E4. 扩展 CPU runner 到更多入口和样例

## Phase F：Triton/GPU 路线

- [x] F1. 设计 `Mini -> Triton/GPU` lowering 路线
- [x] F2. 先支持核心算子 GPU IR lowering
- [x] F3. 再支持 fused op lowering
- [x] F4. 添加 Triton/GPU 路线验证
- [x] F5. 打通 `mini-gpu-lowering` 到 `gpu.launch/gpu.module`
- [ ] F6. 推进云端 A10 上的 `NVVM` / host LLVM 后半链路
- [x] F7. 设计 strategy selection 层，区分 generic GPU / Triton / library-backed 路径
- [x] F8. 为 `linear` / `matmul` / fused op 建立基础策略规则

## Phase G：统一验证

- [ ] G1. Python harness 调用 `compiler-mlir` 工具链
- [ ] G2. 对照 eager / Python reference / MLIR backend
- [ ] G3. 统一 benchmark 入口
- [ ] G4. 统一 artifact dump 入口

## Phase H：Triton 算子作品集文档化

- [x] H1. 新增 `TRITON_OPERATOR_PORTFOLIO.md`
- [x] H2. 明确 5 个作品集入口文件与模块命名约定
- [x] H3. 在 `README.md` 中记录作品集入口文件和运行命令
- [ ] H4. 落地 `triton_matmul.py`
- [ ] H5. 落地 `triton_fused_linear_relu.py`
- [ ] H6. 落地 `triton_softmax.py`
- [ ] H7. 落地 `triton_layernorm.py`
- [ ] H8. 落地 `triton_flash_attention_simplified.py`
- [ ] H9. 为 5 个作品集算子补齐 tests 与 benchmarks

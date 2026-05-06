# 任务清单：Quant Inference Lab

## Phase A：项目初始化

- [x] A1. 创建项目目录
- [x] A2. 编写 README / requirements / design / tasks
- [x] A3. 添加可运行示例
- [x] A4. 添加单元测试

## Phase B：量化核心

- [x] B1. 实现 min-max observer
- [x] B2. 实现对称量化和非对称量化
- [x] B3. 实现 per-tensor 和 per-channel 支持
- [x] B4. 实现参考版 INT8 `linear`
- [ ] B5. 增加 calibration dataset 工具

## Phase C：推理引擎核心

- [x] C1. 建模请求和逐请求指标
- [x] C2. 实现 paged KV allocator 模拟
- [x] C3. 实现 continuous batching 风格调度器
- [x] C4. 添加运行摘要指标
- [ ] C5. 显式拆分 prefill 与 decode 预算

## Phase D：后续扩展

- [ ] D1. 增加 INT4 / group-wise quantization 实验
- [ ] D2. 增加稀疏化实验
- [ ] D3. 增加推理引擎策略对比预设
- [ ] D4. 增加 benchmark 脚本
- [ ] D5. 对接真实推理后端

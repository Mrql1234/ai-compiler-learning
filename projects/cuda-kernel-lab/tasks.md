# CUDA Kernel Lab 任务清单

## Phase 1：项目骨架

- [x] 独立创建 `projects/cuda-kernel-lab`
- [x] 增加 CMake 构建
- [x] 增加 `vector_add` 入口
- [x] 增加 `reduce_sum` 入口
- [x] 增加 `nsys` / `ncu` 脚本
- [x] 补充中文 README

## Phase 2：基础算子

- [ ] 增加 `softmax`
- [ ] 增加 `layernorm`
- [ ] 增加 `sgemm`
- [ ] 给基础算子补 CPU reference 与误差检查

## Phase 3：分析与优化

- [ ] 记录 `nsys` 时间线观察
- [ ] 记录 `ncu` 的 occupancy 与 memory throughput
- [ ] 尝试 shared memory / vectorized load / loop unroll
- [ ] 给每个算子补优化前后对比

## Phase 4：进阶算子

- [ ] 增加 `rope`
- [ ] 增加 `attention`
- [ ] 对比 Triton 与 CUDA 两套实现


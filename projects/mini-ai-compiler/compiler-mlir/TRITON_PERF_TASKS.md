# compiler-mlir Triton 性能落地任务清单

本文档把 [`PERF_MONITORING_PLAN.md`](/home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/PERF_MONITORING_PLAN.md) 里的方向，进一步拆成可以逐步执行的任务列表。

目标不是一次性把 Triton backend 全做完，而是先做出一条**可 benchmark、可 profile、可多轮迭代**的性能主线。

## 1. 里程碑结论

建议按下面 5 个里程碑推进：

1. 建立 Triton 微基线
2. 建立 Triton case / config / 结果归档规范
3. 建立 Triton benchmark + profile 工作流
4. 完成 `fused_linear_relu` 的 2 到 4 轮优化记录
5. 把成熟配置接回 `compiler-mlir` 的 backend selection，并扩展到 `matmul`

其中前 3 个里程碑是“先把工具链搭好”，后 2 个里程碑才是“真正做性能迭代”。

## 2. 推荐新增文件

第一批建议新增或逐步补齐这些文件：

- `scripts/triton_linear_relu_bench.py`
  - Triton `fused_linear_relu` 微基线入口
- `scripts/triton_perf_sweep.py`
  - 扫 `BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages`
- `scripts/triton_profile_iter.py`
  - 统一打包 `nsys` / `ncu` 调用
- `perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json`
  - Triton 主 case
- `perf/configs/triton_linear_relu_a10.json`
  - A10 上的默认配置与 sweep 范围
- `perf/notes/triton_linear_relu_iterations.md`
  - 每轮调优记录
- `perf/CLOUD_TRITON_A10_WORKFLOW.md`
  - 云端 A10 执行清单

第二批再补：

- `scripts/triton_matmul_bench.py`
- `perf/cases/triton_matmul_f32_*.json`
- `perf/configs/triton_matmul_a10.json`
- `perf/notes/triton_matmul_iterations.md`

## 3. 目录约定

从现在开始，Triton 相关产物建议固定落在这些目录：

- `perf/cases/`
  - 输入规模、dtype、layout、correctness 合同
- `perf/configs/`
  - Triton 参数、sweep 空间、profile 目标配置
- `perf/runs/triton_iterations/`
  - benchmark JSON、summary、best config
- `perf/profiles/triton_iterations/`
  - `.nsys-rep`、`.ncu-rep` 及文本摘要
- `perf/notes/`
  - 每轮优化结论和分析记录

这样后面无论是自己复盘，还是面试展示，结构都会非常清楚。

## 4. 里程碑 1：建立 Triton 微基线

### 4.1 目标

先让 `fused_linear_relu` 有一个独立于 `compiler-mlir` 主 runner 的 Triton benchmark 入口。

这个入口的职责非常单纯：

- 生成输入
- 调 Triton kernel
- 做 correctness 对比
- 用统一口径输出 `kernel_ms`

### 4.2 建议实现

建议入口文件：

- `scripts/triton_linear_relu_bench.py`

建议最小功能：

- 支持 `--m --n --k`
- 支持 `--warmup --repeat`
- 支持 `--dtype`
- 支持 `--config`
- 支持 `--json-output`

建议输出字段：

- `case`
- `device`
- `dtype`
- `config`
- `metrics.kernel_ms`
- `metrics.invoke_ms`
- `correctness.max_abs_err`
- `correctness.max_rel_err`

### 4.3 验收标准

- 能在 A10 上稳定跑通
- 能生成 JSON
- 能和 CPU / PyTorch reference 对上结果
- 同一配置重复运行波动在可接受范围

## 5. 里程碑 2：建立 case / config 规范

### 5.1 case 文件

Triton case 要和旧的 baseline case 分开，避免混淆。

建议第一批 case：

- `perf/cases/triton_linear_relu_f32_m128_n128_k128.json`
  - 小 smoke case
- `perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json`
  - 主 benchmark case

建议字段：

- `name`
- `operation`
- `dtype`
- `shape`
- `layout`
- `data_profile`
- `correctness`

### 5.2 config 文件

建议把“当前最佳配置”和“待扫描范围”都写进 config 文件。

例如：

- 默认 config
- tile sweep 范围
- `num_warps` 候选集合
- `num_stages` 候选集合
- profile 时要固定的目标配置

建议入口文件：

- `perf/configs/triton_linear_relu_a10.json`

### 5.3 验收标准

- case 与 config 能解耦
- 改 shape 不需要改 benchmark 代码
- 改 sweep 空间不需要改核心 kernel 文件

## 6. 里程碑 3：建立 benchmark 与 profile 工作流

### 6.1 benchmark 入口

建议新增：

- `scripts/triton_perf_sweep.py`

职责：

- 读 case
- 读 config
- 组合出候选配置
- 调 benchmark
- 输出一轮 sweep 的排序结果

建议命令：

```bash
python3 ./scripts/triton_perf_sweep.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --warmup 10 \
  --repeat 50 \
  --out perf/runs/triton_iterations/iter_01
```

### 6.2 profile 入口

建议新增：

- `scripts/triton_profile_iter.py`

职责：

- 选定一个配置
- 调 `nsys`
- 调 `ncu`
- 导出文本摘要

建议命令：

```bash
python3 ./scripts/triton_profile_iter.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --tag iter_01_best \
  --out perf/profiles/triton_iterations/iter_01_best
```

### 6.3 建议观测指标

benchmark 阶段主要看：

- `kernel_ms`
- `invoke_ms`
- best config 排名

Nsight Compute 阶段主要看：

- achieved occupancy
- SM throughput
- DRAM throughput
- L2 hit rate
- registers per thread
- shared memory per CTA
- stall reason

Nsight Systems 阶段主要看：

- compile
- warmup
- benchmark
- launch
- memcpy

### 6.4 验收标准

- 一轮 sweep 能稳定输出排序结果
- 一组 best config 能自动触发 profile
- profile 结果能落盘到固定目录

## 7. 里程碑 4：完成 `fused_linear_relu` 的多轮优化记录

这是最关键的阶段，也是后面最适合写进项目经历的一段。

### 7.1 Iteration 0

目标：

- 建立最朴素 Triton baseline

需要记录：

- 初始 tile
- 初始 `num_warps`
- 初始 `num_stages`
- 初始 `kernel_ms`
- 第一版 Nsight 观察

### 7.2 Iteration 1

目标：

- 扫 tile

重点记录：

- 哪组 tile 最好
- tile 变大后 occupancy / registers / throughput 怎么变化

### 7.3 Iteration 2

目标：

- 调 `num_warps`
- 调 `num_stages`

重点记录：

- 哪组流水深度更适合 A10
- stall reason 有没有改善

### 7.4 Iteration 3

目标：

- 优化 program mapping
- 观察 L2 和 DRAM 行为

### 7.5 Iteration 4

目标：

- 优化 epilogue 融合写法与写回路径

### 7.6 记录文件

建议用：

- `perf/notes/triton_linear_relu_iterations.md`

每轮固定写四件事：

- 改了什么
- 指标怎么变
- 为什么会这样
- 下一轮准备改什么

## 8. 里程碑 5：接回编译器并扩展到 `matmul`

### 8.1 接回 backend selection

当前先不要一上来就强耦合到 `mini-compiler-gpu-runner`。

更合理的顺序是：

1. 先把 Triton 微基线调成熟
2. 再把“哪些 op / shape 走 Triton”的规则写清楚
3. 最后再接入 `compiler-mlir` 的 backend selection

这一阶段要补的内容包括：

- `mini-triton-prepare`
- `mini-triton-lowering`
- `mini-triton-runtime-lowering`
- `--kernel-backend=triton`

### 8.2 扩展到 `matmul`

当 `fused_linear_relu` 跑顺后，再平移方法到 `matmul`：

- 建立 Triton matmul 微基线
- 重新做 tile / warp / stage sweep
- 用 `cublas` 作为高质量参考 baseline
- 形成第二条可讲清楚的性能故事

## 9. 优先级建议

如果只做最值得投入的前三步，建议顺序是：

1. `scripts/triton_linear_relu_bench.py`
2. `perf/cases/` + `perf/configs/` 的规范化
3. `scripts/triton_perf_sweep.py`

这三步一旦完成，后面就能非常自然地开始“真实的多轮优化”。

## 10. 这份任务清单如何使用

建议实际执行时按下面顺序：

1. 先完成里程碑 1 和 2
2. 再做里程碑 3
3. 然后集中做里程碑 4，形成连续记录
4. 最后再进入里程碑 5

如果后面要把这部分写成项目经历，最有价值的不是“我支持了 Triton”，而是：

> 我把 Triton 路线从无到有搭成了一条可 benchmark、可 profile、可 sweep、可迭代优化的性能工程链路，并在 A10 上围绕 `fused_linear_relu` 和 `matmul` 做了多轮 profile-driven 优化。

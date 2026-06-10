# `compiler-mlir` 大模型算子与 Triton/GPU 路线设计

本文档记录 `projects/mini-ai-compiler/compiler-mlir/` 面向更真实大模型工作负载的扩展设计，重点回答两件事：

- 如何把当前以 `MLP / linear + relu` 为主的教学型项目，扩展到更贴近大模型推理的核心算子集合
- 如何在当前 `generated_nvvm / cuda_hand / cublas` 基础上引入 Triton，并先聚焦 GPU 后端形成可执行、可分析、可优化的路线

当前文档是设计文档，不代表所有能力已实现。

---

## 1. 设计目标

### 1.1 目标

把当前项目从“小型静态教学 case”升级为“单层或单块级真实工作负载编译器实验平台”，满足以下目标：

- 支持更接近大模型推理的核心算子
- 支持从高层 `mini` IR 到 GPU 后端的多分叉 lowering
- 支持 Triton 作为热点融合 kernel 的专用后端
- 支持对自动生成 kernel、Triton kernel、库 kernel 做统一性能分析
- 支持后续引入动态 shape、KV cache、decode/prefill 分离优化

### 1.2 非目标

当前阶段不追求：

- 一次性支持完整大模型端到端编译
- 一次性支持 GPU、AMD、NPU 等所有硬件后端
- 一次性覆盖训练图、反向传播和自动求导
- 一次性做完整的图级自动并行、张量并行、流水并行

当前阶段只先聚焦：

- 推理场景
- 单卡 GPU
- 单个 decoder block 或局部子图
- correctness、lowering、profiling、性能迭代闭环

---

## 2. 当前项目边界

当前 `compiler-mlir` 已支持的 `mini` op 主要有：

- `mini.constant`
- `mini.linear`
- `mini.qlinear`
- `mini.matmul`
- `mini.add`
- `mini.relu`
- `mini.fused_linear_relu`
- `mini.qlinear_relu`
- `mini.fused_matmul_add_relu`

当前项目已经具备：

- `mini -> linalg/tensor/arith`
- `mini -> gpu.launch_func / gpu.module`
- `gpu -> nvvm`
- `generated_nvvm / cuda_hand / cublas` 三路线性能对比框架

当前项目还明显缺少：

- attention、softmax、normalization、rope 等核心算子
- 动态 shape 支持
- KV cache 的 IR 与 runtime 契约
- block 级真实 workload
- Triton 专用 lowering 分支
- library backend 的系统化调优参数模型

因此后续扩展必须明确遵循“先 block，后整模；先 GPU，后多后端”的节奏。

---

## 3. 推荐目标工作负载

### 3.1 为什么不直接做完整大模型

完整大模型编译会同时引入过多问题：

- 前端导入复杂度
- 动态 shape 与运行时调度
- KV cache 生命周期
- attention 子图分区
- 多类融合策略
- 后端差异与算子覆盖率
- correctness 验证范围过大

对当前项目来说，这样会把“编译器结构演进”和“系统集成复杂度”混在一起，不利于持续推进。

### 3.2 推荐真实任务

推荐把目标收敛为：

- `decoder block`
- `attention block`
- `mlp block`

建议阶段性 case：

1. `RMSNorm + Linear + SiLU/GELU + Linear`
2. `RoPE + QKV projection + attention core`
3. `prefill attention block`
4. `decode attention block with KV cache`

这样既比当前 `MLP` 更贴近真实大模型，又不会一开始就把问题规模拉到不可控。

---

## 4. 算子扩展范围

### 4.1 第一批必须支持的算子

为了覆盖主流 decoder-only 大模型，建议先加入以下 `mini` 高层算子：

- `mini.rmsnorm`
- `mini.softmax`
- `mini.rope`
- `mini.transpose`
- `mini.reshape`
- `mini.concat`
- `mini.slice`
- `mini.gather`
- `mini.silu` 或 `mini.gelu`
- `mini.attention` 或 `mini.sdpa`

如果只做分解式 attention，也至少要有：

- `matmul`
- `add`
- `softmax`
- `transpose`
- `reshape`
- `slice`

### 4.2 第二批建议支持的算子

- `mini.layernorm`
- `mini.mul`
- `mini.div`
- `mini.cast`
- `mini.where`
- `mini.masked_fill`
- `mini.reduce_max`
- `mini.reduce_sum`

### 4.3 KV cache 相关能力

KV cache 不建议一开始伪装成普通纯函数 op。更合理的做法是分成两层：

- IR 层描述 cache 访问语义
- runtime 层管理 cache buffer 生命周期

建议后续引入以下能力：

- `mini.kv_read`
- `mini.kv_write`
- `mini.kv_append`
- `mini.kv_view`

也可以先不暴露太多专用 op，而是先在 block 级 runner 中约定 cache ABI，再逐步内化到 `mini` dialect。

---

## 5. 动态 shape 设计方向

### 5.1 为什么动态 shape 是必须项

大模型推理几乎天然依赖动态 shape：

- batch size 变化
- prefill 阶段 `seq_len` 变化
- decode 阶段 `kv_len` 递增
- 不同请求混合调度

如果只支持静态 shape，很多真实推理场景无法表达。

### 5.2 推荐分阶段支持方式

第一阶段：

- op verifier 允许部分维度动态
- 仍优先支持 rank 固定
- 运行时显式传 `m/n/k/seq_len/head_dim/kv_len`

第二阶段：

- 补 shape inference / reification
- 支持 bufferization 中的动态维分配
- runtime ABI 显式传 sizes / strides

第三阶段：

- 在 profiling case 中支持 shape sweep
- 对不同 shape 桶做独立 benchmark
- 为 Triton / 库 backend 提供 shape-specialized 调优配置

### 5.3 当前建议

不要一开始就追求“全动态图”。先支持：

- `batch=1`
- `num_heads` 固定
- `head_dim` 固定
- `seq_len` / `kv_len` 动态

这已经足够覆盖大量推理优化实验。

---

## 6. IR 分层建议

### 6.1 High-Level `mini` 层

这一层保留模型语义，便于：

- 图级融合
- 算子分区
- backend 选择
- correctness 对照

这一层建议长期保留的核心 op：

- `linear`
- `matmul`
- `softmax`
- `rmsnorm`
- `rope`
- `attention`
- `reshape/transpose/slice/gather`
- `kv_*`

### 6.2 Standard Tensor / Linalg 层

把 `mini` 语义 lower 成：

- `arith`
- `tensor`
- `linalg`
- `scf`
- `func`

这一层适合做：

- decomposition
- 通用 canonicalization
- tile-friendly rewrite
- 一部分通用 fusion

### 6.3 Backend Split 层

这里是关键分叉点。建议按热点子图而不是整个 module 分叉。

推荐分叉模型：

```text
mini high-level op / fused subgraph
-> backend selector
-> generated_nvvm
-> triton
-> library
-> hand kernel
```

注意：

- backend 选择应基于“子图模式 + shape + dtype + layout”
- 不同 backend 共享同一份输入、输出和 correctness 协议

---

## 7. Triton 在体系中的角色

### 7.1 Triton 不是通用总后端

在这个项目里，Triton 的角色不是取代整个 MLIR GPU 路线，也不是取代库实现。

更准确地说，Triton 是：

- GPU 热点 kernel DSL
- 融合 kernel 候选后端
- 介于“完全自动生成”和“手写 CUDA”之间的专用路线

### 7.2 Triton 适合承接的算子

优先考虑以下类型：

- fused elementwise
- fused matmul epilogue
- attention 内部的局部融合 kernel
- RMSNorm / RoPE / softmax 这类需要较强 kernel 级控制的热点

不建议第一阶段就用 Triton 接管：

- 全部图执行
- 复杂 runtime 生命周期管理
- KV cache 完整系统

### 7.3 Triton 与其他 GPU 路线的分工

- `generated_nvvm`
  - 作为通用 MLIR GPU baseline
  - 覆盖范围广
  - 便于验证公共 lowering 结构

- `triton`
  - 作为热点 kernel specialized 路线
  - 用于提升融合 kernel 性能
  - 适合作为自动生成高性能 kernel 的重点候选

- `library`
  - 用于大规模标准算子
  - 例如大 `matmul`、大 `linear`、标准 attention primitive

- `cuda_hand`
  - 用于教学、对照和极端定制场景
  - 不应成为项目长期默认主线

---

## 8. GPU First 后端策略

当前先只做 GPU，推荐分三条执行路线：

### 8.1 Route A: `generated_nvvm`

链路：

```text
mini
-> linalg/tensor
-> bufferization
-> gpu dialect
-> nvvm
-> ptx/cubin/fatbin
-> MLIR GPU runtime execute
```

用途：

- 通用自动生成 baseline
- 对比 Triton / 库实现
- 检查公共 lowering 是否正确

### 8.2 Route B: `triton`

链路：

```text
mini fused subgraph
-> triton-oriented lowering
-> Triton kernel IR / source
-> Triton compile
-> GPU kernel launch
```

用途：

- 生成更有针对性的热点 kernel
- 在保持较高开发效率的同时追求更强性能

### 8.3 Route C: `library`

链路：

```text
mini fused subgraph
-> runtime-call lowering
-> cublas / cutlass / vendor library
-> optional epilogue kernel
```

用途：

- 大规模标准算子的强 baseline
- 大 `matmul` / `linear` 的默认优先路线

---

## 9. Backend 选择规则

建议第一版采用规则式选择，而不是一开始做复杂 cost model。

### 9.1 推荐初始规则

- `large matmul / linear`
  - 默认优先 `library`

- `small-to-medium fused epilogue`
  - 默认优先 `triton`

- `simple elementwise / correctness baseline`
  - 默认优先 `generated_nvvm`

- `教学对照 / 定制实验`
  - 使用 `cuda_hand`

### 9.2 后续再引入的决策因子

- shape bucket
- dtype
- layout
- 是否有动态维度
- workspace 可用性
- 是否需要 KV cache 读写
- 是否需要多 kernel 融合

---

## 10. Runtime ABI 与执行模型

### 10.1 统一执行契约

无论走哪条路线，都应共享：

- 同一份输入语义
- 同一份输出 buffer 语义
- 同一份 correctness 检查
- 同一份 profiling 契约

### 10.2 面向 block 级 workload 的运行时需求

为了支持 attention 和 KV cache，runner 后续需要补充：

- block 级输入生成
- 多输入张量
- cache buffer 构造与回收
- prefill / decode 模式切换
- full-output correctness 校验

### 10.3 当前建议

当前先不要尝试把所有 runtime 问题塞进 `func.call` ABI。更合理的推进方式是：

1. 先用 runner-level problem object 表达 block 输入
2. 对局部热点子图继续保留 runtime-call lowering
3. 等 block 级路径稳定后，再决定哪些 ABI 进入正式 lowering

---

## 11. Profiling 与优化闭环

### 11.1 主指标

继续沿用当前方案：

- 主指标：`kernel_ms`
- 辅助指标：`invoke_ms`、`compile_ms`、`prepare_ms`、`end_to_end_ms`

### 11.2 新 workload 的 profiling 重点

#### RMSNorm / RoPE

- memory bandwidth
- vectorization
- register pressure
- launch overhead

#### Attention

- QK matmul
- softmax
- V matmul
- cache read/write
- layout transform

#### MLP block

- large matmul / linear
- activation fusion
- epilogue kernel 访存

### 11.3 三路线统一对比

对每个 block 级 case，统一比较：

- `generated_nvvm`
- `triton`
- `library`

必要时再附加：

- `cuda_hand`

### 11.4 当前工具延续方式

当前已有工具可以继续复用：

- `mini-compiler-gpu-runner`
- `scripts/perf_run.py`
- `scripts/perf_compare.py`
- `scripts/perf_profile_nsys.sh`
- `scripts/perf_profile_ncu.sh`

后续只需要把 case 从 `linear_relu` 扩展到 block 级 workload。

---

## 12. 推荐实施顺序

### Phase 1: Block 级语义扩展

- 增加 `mini.rmsnorm`
- 增加 `mini.rope`
- 增加 `mini.softmax`
- 增加必要的 `reshape/transpose/slice/gather`

目标：

- 能表达真实 attention / mlp block

### Phase 2: 静态 shape block GPU 路线

- 先支持固定 `head_dim`
- 先支持固定 `num_heads`
- 支持固定 shape 的 attention / mlp block

目标：

- 跑通 GPU correctness 和 profiling

### Phase 3: Triton 分支接入

- 为 RMSNorm / RoPE / softmax / fused epilogue 提供 Triton backend
- 让 `backend selector` 能在 `generated_nvvm` 和 `triton` 之间切换

目标：

- 形成可比较的自动生成 GPU 双路线

### Phase 4: 动态 shape 与 KV cache

- 支持动态 `seq_len`
- 支持动态 `kv_len`
- 引入 prefill / decode 两类 case
- 引入 cache runtime object

目标：

- 更贴近真实大模型推理

### Phase 5: 更大 workload 与性能调优

- decoder block 级 benchmark
- backend-specific tuning config
- shape bucket profiling
- Triton / library / generated kernel 的策略选择

目标：

- 形成持续优化闭环

---

## 13. 建议的入口文件与当前命令

虽然本文档描述的是未来设计，但当前建议从以下入口继续推进：

### 13.1 入口文件

- `LARGE_MODEL_GPU_DESIGN.md`
- `LOWERING_ROADMAP.md`
- `PERF_MONITORING_PLAN.md`
- `test/gpu_prep.mlir`
- `test/gpu_runner_demo.mlir`
- `perf/cases/gpu_runner_demo.json`

### 13.2 当前可执行命令

查看当前 GPU lowering 主线：

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
```

查看当前 runtime-call lowering：

```bash
./build/bin/mini-compiler-opt test/gpu_runtime_call_lowering.mlir \
  --pass-pipeline='builtin.module(func.func(mini-canonicalize,mini-fusion),mini-gpu-runtime-call-lowering{backend=cublas})'
```

运行当前 GPU runner：

```bash
./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
  --kernel-backend=generated_nvvm
```

运行当前性能对比：

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend cublas \
  --metric kernel_ms
```

这些命令目前还不是“大模型 block 路线”的最终入口，但它们是后续扩展最直接的起点。

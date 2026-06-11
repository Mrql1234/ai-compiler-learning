# compiler-mlir GPU 性能迭代方案（Triton 主线）

本文档用于重新定义 `projects/mini-ai-compiler/compiler-mlir/` 的 GPU 性能工作重点。

这次重写的核心目标不是继续维护“`generated_nvvm` / 手写 CUDA / `cuBLAS` 三路线横向排行榜”，而是建立一条更适合 AI 编译器项目长期演进的单主线优化方案：**以 Triton 为主要迭代对象，在真实 A10 GPU 上围绕一个 kernel 连续做多轮可解释优化**。

## 1. 为什么要重做

现有 `PERF_MONITORING_PLAN.md` 有三个明显问题：

- 它把“谁更快”放在第一位，重心是三条路线横向比较，而不是“这一条路线为什么会变快”。
- 它把手写 CUDA 放成长期路线之一，但对当前项目目标来说，手写 CUDA 更适合教学和局部验证，不适合作为后续主线。
- 它没有把“改了什么参数 / 写法 -> 会影响什么硬件行为 -> 应该看哪些指标 -> 如何根据指标决定下一轮修改”写成明确闭环。

所以新的结论是：

- **主迭代路线选 Triton**
- **主展示对象先选 `fused_linear_relu`，再扩展到 `matmul`**
- **`generated_nvvm` 和 `library` 只保留为阶段性参考基线，不再作为日常并行优化主线**

## 2. 新方案的结论

### 2.1 路线选择

新的 GPU 性能工作流采用“**单主线迭代 + 低频参考对比**”：

- 主线：`triton`
- 参考基线 1：`generated_nvvm`
- 参考基线 2：`library`（对 `matmul` 主要指 `cublas`，未来可扩展 `cutlass`）

不再把 `cuda_hand` 作为正式长期路线。

### 2.2 算子选择

推荐分两阶段推进：

1. 第一阶段先把 `mini.fused_linear_relu` 做成完整的 Triton 迭代优化案例
2. 第二阶段再把同一套方法迁移到 `mini.matmul`

这样安排的原因是：

- `fused_linear_relu` 更符合 Triton 在项目中的角色：热点融合 kernel、可控、可解释、便于展示“融合前后访存变化”
- `matmul` 当然重要，但第一版就直接做高质量 Triton matmul，复杂度会明显更高，容易把精力都耗在大块模板和流水细节里
- 先把 `fused_linear_relu` 做扎实，更容易形成一套清楚的性能方法论；随后迁移到 `matmul` 时，面试里也更容易讲“方法迁移”

### 2.3 性能目标

新的重点不是每次都同时比较三条路线，而是：

- 先在 Triton 路线内部做 2 到 4 轮连续优化
- 每轮都记录“修改点 - 指标变化 - 原因判断 - 下一步动作”
- 在阶段收尾时，再拿 `generated_nvvm` 或 `cublas` 做一次参考对照

也就是说，**比较只是收尾动作，迭代过程才是主角**。

## 3. 为什么主线应该选 Triton

对当前项目，Triton 比另外两类路线更适合作为性能迭代主线。

### 3.1 相比 `generated_nvvm`

`generated_nvvm` 的价值主要在于：

- 验证 MLIR lowering 是否正确
- 提供通用 GPU baseline
- 观察从 `mini -> linalg/scf/gpu/nvvm` 的编译链路

但如果把它作为性能主线，会有两个问题：

- 可调空间分散在 pass、tile、map、bufferization、launch 结构等多个层次，单轮迭代反馈链偏长
- 很多优化结论不够“就地可解释”，更像在调编译链全局行为

它适合做“通用 baseline”和“编译器结构验证”，不适合做当前阶段最主要的性能迭代故事。

### 3.2 相比 `library`

`cublas` 这类库路线适合作为高质量参考上界，但不适合作为主要优化对象：

- 黑盒程度高
- 可调参数有限
- 很难直接体现“你改了编译器 / kernel 生成策略之后，性能为什么变化”

它适合回答“你离成熟库还有多远”，不适合回答“你是如何做性能工程的”。

### 3.3 相比手写 CUDA

手写 CUDA 的问题不是“不能快”，而是它不再适合这个项目的长期定位：

- 开发重心会偏到底层细节实现，而不是编译器可生成、可迁移、可自动调参的路线
- agent 或编译器自动生成 Triton kernel，比自动写高质量 CUDA 更现实
- Triton 的参数面更自然地适合做“生成候选实现 -> 跑 benchmark -> 保留最优配置”的闭环

因此，对“AI 编译器项目如何继续深化”这个问题，Triton 是更合理的主线。

## 4. 当前仓库基础与可行性判断

结论：**可行，但 `compiler-mlir` 里当前还没有真正落地 Triton backend 实装，所以这份文档描述的是下一阶段性能迭代方案，而不是现成可执行能力。**

当前已经具备的基础有：

- `mini.fused_linear_relu` 这类高层融合 op 已存在
- A10 云端验证链路、benchmark 脚本、Nsight 工作流已经有第一版积累
- `generated_nvvm` 路线已经能作为通用 GPU baseline
- `cublas` 路线已经能作为 `linear_relu` / `matmul` 类问题的参考 baseline
- `LOWERING_ROADMAP.md` 和 `LARGE_MODEL_GPU_DESIGN.md` 已明确给出 Triton 分支的设计方向

当前缺的关键环节有：

- `compiler-mlir` 内真正可运行的 Triton kernel 生成与 launch 入口
- Triton 专用 case / config / profiling 入口
- 面向迭代优化的数据归档约定

所以新的性能路线建议分两层：

1. 先补 Triton 微基线与调优入口
2. 再把成熟的 Triton kernel 方案接回 `compiler-mlir` 的 backend selection

## 5. 新的工作结构

## 5.1 主结构

新的性能工作流只保留两层：

- **Triton 微基线**
  - 用于快速改 kernel、扫参数、看 Nsight Compute
  - 这是日常迭代的主战场
- **编译器集成校验**
  - 用于验证 `mini` 高层 op 或子图是否能稳定映射到 Triton 路线
  - 只在阶段性里程碑运行，不作为每轮调参主入口

### 5.2 参考路线的角色

- `generated_nvvm`
  - 用来回答“通用 MLIR GPU 路线现在做到什么水平”
- `library`
  - 用来回答“成熟库路线的参考上界在哪里”
- `triton`
  - 用来回答“我们这一轮到底是怎么把 kernel 做快的”

## 6. 优先优化哪个算子

### 6.1 第一优先：`fused_linear_relu`

第一阶段优先选 `fused_linear_relu`，理由是：

- 它天然带有融合价值，能讲清楚“减少中间结果落盘”和“降低额外 memory traffic”
- 它比完整 `matmul` 更容易收敛出清晰的第一版优化故事
- 它正好对应当前项目中已经存在的融合算子，不是凭空换题

第一阶段的目标不是追平 cuBLAS，而是：

- 做出稳定可运行的 Triton fused kernel
- 形成 2 到 4 轮清晰迭代记录
- 用指标证明优化方向是否正确

### 6.2 第二优先：`matmul`

`matmul` 作为第二阶段延伸更合适，原因是：

- `matmul` 是 AI 编译器岗位的核心算子，后面一定要做
- 但 Triton `matmul` 的 tile、pipeline、数据复用、甚至 split-K 等问题更复杂
- 先在 `fused_linear_relu` 上打通迭代方法，再迁移到 `matmul`，会更稳

## 7. 迭代优化必须固定的实验合同

如果不固定实验条件，任何性能变化都不可信。每一轮迭代都必须固定：

- 设备：云端 NVIDIA A10，`sm_86`
- dtype：第一阶段固定 `f32`
- 输入布局：固定 row-major 约定
- case 规模：至少固定 1 个主 case，建议保留 1 个小 smoke case + 1 个大 case
- warmup / repeat：例如 `warmup=10`，`repeat=50`
- correctness 容忍度：固定 `atol` / `rtol`
- 计时口径：默认只看 steady-state `kernel_ms`

建议第一阶段固定两个 case：

- 小 case：用于 correctness 和快速回归
- 大 case：用于真实吞吐观察

对 `fused_linear_relu`，建议主 case 沿用或扩展当前大 shape，例如：

- `M=1024, N=1024, K=1024`

## 8. Triton 主线要记录哪些产物

每一轮迭代都应至少留下这些产物：

- Triton kernel 源码或生成模板
- 本轮配置
  - 例如 `BLOCK_M/BLOCK_N/BLOCK_K`
  - `num_warps`
  - `num_stages`
  - 是否 grouped ordering
  - 是否做 epilogue fusion
- benchmark JSON
- Nsight Compute 摘要
- 一小段结论记录
  - 改了什么
  - 哪个指标变了
  - 推断原因
  - 下一轮准备怎么改

建议后续统一归档到类似目录：

- `perf/cases/`
- `perf/runs/triton_iterations/`
- `perf/profiles/triton_iterations/`
- `perf/notes/`

当前这些目录不要求已经全部存在，但设计上应按这个思路落盘。

## 9. 重点参数及其影响

这一节是整个方案的核心。每次改参数，都要知道它为什么可能有效。

### 9.1 `BLOCK_M / BLOCK_N / BLOCK_K`

这是 Triton kernel 最直接的一组 tile 参数。

- `BLOCK_M`
  - 控制单个 program 负责多少行
- `BLOCK_N`
  - 控制单个 program 负责多少列
- `BLOCK_K`
  - 控制 K 维分块大小

它们会直接影响：

- 单次计算的数据复用
- shared memory / register 压力
- global memory 访问模式
- 程序并行度
- mask 边界开销

常见趋势是：

- tile 太小：并行度高，但复用差，launch 数量多，访存效率低
- tile 太大：复用更好，但资源占用变高，可能导致 occupancy 降低

### 9.2 `num_warps`

`num_warps` 表示一个 Triton program 使用多少个 warp 协同执行。

它主要影响：

- 一个 tile 内部的并行度
- latency hiding 能力
- 每个线程块的资源占用

常见趋势是：

- 太小：单 tile 计算资源不足，吞吐上不去
- 太大：寄存器压力上升，occupancy 下降，反而变慢

### 9.3 `num_stages`

`num_stages` 是 Triton 中很关键、也很适合面试讲清楚的参数。

它表示 K-loop 软件流水的 stage 深度，可以理解为：

- 在当前计算进行时，提前准备后续迭代需要的数据
- 用更多流水级去隐藏 global memory latency

它主要影响：

- latency hiding
- shared memory / register 占用
- pipeline 深度

常见趋势是：

- `num_stages` 偏小：流水浅，访存延迟隐藏不够
- `num_stages` 增大：可能提升吞吐，但也会增加资源占用
- 增到一定程度后：收益变小，甚至因 occupancy 降低而变差

所以 `num_stages` 不是越大越好，它本质上是“延迟隐藏”和“资源压力”之间的权衡参数。

### 9.4 grouped ordering / program mapping

对于 `matmul` 或 `linear` 类 kernel，program 的遍历顺序会影响：

- L2 cache reuse
- 权重块重复访问的局部性
- 不同 CTA 之间的空间局部性

例如：

- 朴素按行列线性展开
- 按 `GROUP_M` 做 grouped ordering

这类调整往往不改变数学计算，但会改变 cache 命中情况和整体 memory behavior。

### 9.5 epilogue 融合写法

对 `fused_linear_relu`，最关键的一类变化不是算得更快，而是**少搬一次数据**。

需要重点比较：

- matmul 结果先写回，再单独做 bias/relu
- matmul 累加结束后直接做 bias + relu，再一次性写回

它会影响：

- global memory 写回次数
- launch 次数
- 中间张量大小
- DRAM throughput 压力

这一点非常适合拿来讲“融合算子为什么值钱”。

### 9.6 load/store 与 layout 写法

同样的数学逻辑，不同的地址计算和 load/store 写法也会影响性能，例如：

- 哪个维度连续
- 是否便于 coalesced access
- 是否过多使用 mask
- 是否能做更规整的向量化访存

这些修改通常会反映在：

- DRAM 吞吐
- L2 hit rate
- memory stall reason

## 10. 需要观测哪些指标

### 10.1 第一层：结果指标

- `correctness`
  - 结果是否正确
- `kernel_ms`
  - 核心主指标
  - steady-state kernel 时间
- `invoke_ms`
  - host 侧单次调用时间
  - 用来看 launch 或 runtime 包装开销

其中排序和阶段收敛主要看：

- `kernel_ms.median`

### 10.2 第二层：吞吐指标

对 `fused_linear_relu` 建议额外看：

- effective GFLOPS / TFLOPS
- effective GB/s

它们的作用是：

- 如果 `kernel_ms` 变好了，吞吐也应该同步改善
- 如果时间变好了但吞吐解释不通，说明可能有计时口径或 workload 变化

### 10.3 第三层：Nsight Compute 指标

建议优先看以下几类：

- achieved occupancy
  - 当前活跃 warp 比例
- SM throughput
  - 计算单元忙碌程度
- DRAM throughput
  - 外存带宽利用率
- L2 throughput / hit rate
  - cache 行为
- registers per thread
  - 寄存器压力
- shared memory per CTA
  - 共享内存压力
- warp stall reasons
  - 例如 memory dependency、long scoreboard、not selected 等

这些指标的解释重点是：

- occupancy 低，不一定一定慢，但往往说明 latency hiding 空间不足
- DRAM throughput 很高而 SM throughput 上不去，通常更像 memory-bound
- SM throughput 高、DRAM 不是瓶颈，但 `kernel_ms` 仍慢，可能是并行度、tile 或 pipeline 没吃满
- registers / shared memory 过高，会压低 occupancy

### 10.4 第四层：Nsight Systems 指标

Nsight Systems 不是用来看 kernel 内部细节，而是用来看阶段边界：

- compile
- warmup
- benchmark
- kernel launch
- memcpy

在 Triton 主线里，Nsight Systems 主要回答：

- 是 kernel 本身慢，还是 launch / runtime / compile 干扰大

## 11. 推荐的详细迭代过程

下面给出推荐的第一版详细流程。这个部分正是简历和面试里最值钱的内容。

### Iteration 0：建立最朴素 Triton baseline

目标：

- 先有一个正确、可跑、可测的 Triton `fused_linear_relu`
- 不追求最优，只追求有稳定 baseline

建议做法：

- 使用简单 tile
- 使用固定 `num_warps`
- 使用较保守 `num_stages`
- 先不引入复杂 reorder

这一轮主要看：

- correctness
- `kernel_ms`
- occupancy
- DRAM throughput
- 主要 stall reason

预期现象：

- 性能通常不会太好
- 但能帮助确认它更偏 memory-bound 还是 compute-bound

### Iteration 1：扫 tile 大小

修改点：

- 调 `BLOCK_M`
- 调 `BLOCK_N`
- 调 `BLOCK_K`

为什么先做这个：

- tile 是最直接决定数据复用、并行度和边界 mask 比例的参数
- 它对性能影响通常最大，也最容易形成清晰结论

重点观察：

- `kernel_ms`
- effective throughput
- occupancy
- registers / shared memory

如何判断结果：

- 如果 tile 变大后 `kernel_ms` 下降，同时 occupancy 还能接受，说明复用提升带来了收益
- 如果 tile 继续变大后反而变慢，且 registers 或 shared memory 明显升高，说明资源压力开始反噬

### Iteration 2：调 `num_warps` 和 `num_stages`

修改点：

- 固定较优 tile 后，扫描 `num_warps`
- 再扫描 `num_stages`

为什么这一步放在 tile 之后：

- `num_warps` 和 `num_stages` 的有效区间依赖 tile 选择
- 先把 tile 调到合理区间，再看 pipeline / 并行度更容易收敛

重点观察：

- `kernel_ms`
- achieved occupancy
- stall reason
- register pressure

如何解释：

- `num_warps` 增大后如果吞吐提升，说明单 tile 内并行度原先不足
- `num_warps` 增大后如果 occupancy 大降，说明资源占用过高
- `num_stages` 增大后如果 long scoreboard stall 降低，通常表示 latency hiding 更好
- `num_stages` 增大后如果收益不再增加，甚至变慢，通常是资源占用吃掉了好处

### Iteration 3：优化 program mapping 与 cache 行为

修改点：

- 引入 grouped ordering
- 调整 program id 到 tile 的映射方式

这一步主要解决：

- 相邻 program 对权重块或输入块的复用不足
- L2 行为不稳定

重点观察：

- `kernel_ms`
- L2 hit rate
- DRAM throughput
- memory-related stall

如何解释：

- 如果 `kernel_ms` 明显改善，但 arithmetic 配置没变，通常说明 cache 行为变好了
- 如果 L2 指标没改善，时间也没改善，这类 reorder 就没有价值

### Iteration 4：优化 epilogue 融合与写回方式

修改点：

- bias + relu 在 accumulator 上直接完成
- 避免中间结果单独落盘
- 优化 store 写回路径和 mask 处理

这一步为什么重要：

- `fused_linear_relu` 的核心价值不只是“算得快”，更是“少一次写回，再少一次读回”

重点观察：

- `kernel_ms`
- DRAM throughput
- store 相关 stall
- launch 次数

如何解释：

- 如果融合后 `kernel_ms` 降低、DRAM 压力下降，说明融合真正减少了 memory traffic
- 如果融合后时间变化不明显，说明瓶颈可能还在主 matmul 计算而不是 epilogue

### Iteration 5：固化 shape-specialized 配置

修改点：

- 为 A10 上常见 shape 保留一组或多组已验证配置
- 必要时做规则式 config 选择

目标：

- 不再只是一组“碰巧快”的参数
- 而是形成稳定、可复用、可解释的 shape bucket 策略

这一步的输出应该是：

- 哪些 shape 用哪组 tile / warp / stage
- 为什么这么选
- 和 `generated_nvvm` / `cublas` 的阶段性差距

## 12. 什么时候再去看其他路线

新的流程里，不应该每改一点就重新跑三路线大对比。

更合理的节奏是：

- Triton 内部连续做 2 到 3 轮优化
- 形成稳定改进后，再做一次阶段性对比

建议的对比频率：

- `generated_nvvm`
  - 每个大里程碑跑一次
- `cublas`
  - `matmul` 里程碑跑一次
- Triton 自身 sweep
  - 每轮都跑

这样才能把注意力放回“优化过程”，而不是“排行榜截图”。

## 13. 当前仓库里的入口文件

本次文档重写后，相关入口建议按下面理解：

### 13.1 当前已存在且可执行

- `LARGE_MODEL_GPU_DESIGN.md`
  - Triton 在整体后端体系里的角色
- `LOWERING_ROADMAP.md`
  - Triton 路线与通用 GPU 路线的分层设计
- `scripts/perf_run.py`
  - 当前可执行的性能运行入口
- `scripts/perf_compare.py`
  - 当前结果汇总入口
- `scripts/perf_profile_nsys.sh`
  - Nsight Systems 入口
- `scripts/perf_profile_ncu.sh`
  - Nsight Compute 入口
- `perf/cases/gpu_runner_demo.json`
  - 当前小型归档 case
- `perf/cases/linear_relu_f32_m1024_n1024_k1024.json`
  - 当前大 case 参考
- `TRITON_PERF_TASKS.md`
  - Triton 主线的实际落地任务清单
- `perf/configs/README.md`
  - Triton config 目录约定
- `perf/notes/README.md`
  - Triton 迭代记录目录约定

### 13.2 当前设计上应新增

- Triton 微基线运行入口
- Triton 参数 sweep 入口
- Triton profile 结果归档约定
- Triton config 文件

这一部分是后续实现任务，不是当前仓库已完成能力。

## 14. 当前可执行命令

虽然 Triton 路线还没在 `compiler-mlir` 内落地，但当前仍可用已有命令做准备工作和参考观测。

### 14.1 查看当前 GPU lowering

```bash
./build/bin/mini-compiler-opt --mini-gpu-lowering test/gpu_prep.mlir
```

### 14.2 运行当前 `generated_nvvm` 基线

```bash
./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
  --kernel-backend=generated_nvvm \
  --warmup=10 \
  --repeat=50
```

### 14.3 运行当前归档 case 的参考对比

```bash
python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cublas \
  --metric kernel_ms \
  --warmup 10 \
  --repeat 50 \
  --run-dir perf/runs/gpu_runner_demo_reference
```

```bash
python3 ./scripts/perf_compare.py \
  --metric kernel_ms \
  perf/runs/gpu_runner_demo_reference/summary.json
```

### 14.4 采集当前 baseline 的 Nsight 报告

```bash
./scripts/perf_profile_nsys.sh \
  perf/profiles/gpu_runner_demo_mlir_nvvm_nsys \
  ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
    --warmup=1 \
    --repeat=2 \
    --cubin-format=fatbin
```

## 15. 建议新增的 Triton 命令形态

下面这些命令是**后续建议实现的入口形态**，当前仓库还没有对应脚本：

```bash
python3 ./scripts/triton_perf_sweep.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --warmup 10 \
  --repeat 50 \
  --out perf/runs/triton_iterations/iter_01
```

```bash
python3 ./scripts/triton_profile_iter.py \
  --case perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json \
  --config perf/configs/triton_linear_relu_a10.json \
  --out perf/profiles/triton_iterations/iter_01
```

这里的重点不是脚本名字本身，而是入口应支持：

- case
- config
- benchmark
- profile
- 结果归档

## 16. 最终建议

新的性能叙事应该改成下面这句话：

> 在 A10 上围绕 Triton `fused_linear_relu` kernel 做多轮 profile-driven 优化，连续调整 tile、program mapping、`num_warps`、`num_stages` 与 epilogue 融合写法，并结合 `kernel_ms`、occupancy、DRAM/L2 throughput、stall reason 等指标完成迭代收敛；最终再以 `generated_nvvm` 和 `cublas` 作为阶段性参考基线。

这比“我比较过三条路线谁快”更像真正的 AI 编译器性能工程。

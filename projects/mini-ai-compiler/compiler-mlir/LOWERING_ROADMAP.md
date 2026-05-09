# `compiler-mlir` Lowering Roadmap

本文档描述 `projects/mini-ai-compiler/compiler-mlir/` 的 lowering 设计思路、推荐分层，以及 CPU / GPU / Triton / 国内厂商芯片等多后端路线。

目标不是“每个后端各写一条完全独立的链路”，而是：

- 尽量复用公共 lowering
- 在关键分叉点接入后端特定 pass
- 让 CPU 成为最先跑通的正式主线
- 让 Triton / GPU / vendor backend 成为后续扩展方向

---

## 1. 设计原则

### 1.1 分层 lowering，而不是一步到位

推荐始终遵循下面的层次：

1. `mini` 高层语义
2. 标准张量 / 结构化计算语义
3. buffer / memory 语义
4. loop / control-flow 语义
5. LLVM / GPU / vendor runtime 语义

也就是说：

- 先消灭自定义高层算子
- 再把 tensor 语义转成 buffer 语义
- 再把结构化算子拆成 loops / cfg
- 最后才 lower 到目标后端

### 1.2 公共主线和后端分叉分开

不要让每个后端都维护一整套完全独立的 lowering 链路。

推荐组织方式：

- 公共 lowering：
  - `mini -> linalg/tensor/arith`
  - canonicalize / fold / fusion
  - bufferization 前准备
- 后端分叉 lowering：
  - CPU
  - Triton
  - 通用 GPU dialect
  - vendor backend
- runtime / executable lowering：
  - LLVM IR / JIT / runner
  - kernel launch
  - graph runtime / vendor runtime

### 1.3 优先复用官方 pass，差异点再自定义

应优先复用：

- `one-shot-bufferize`
- `convert-bufferization-to-memref`
- `convert-linalg-to-loops`
- `convert-scf-to-cf`
- `convert-cf-to-llvm`
- `convert-func-to-llvm`
- `mlir-translate --mlir-to-llvmir`

更可能需要自定义：

- `mini` dialect 到标准 dialect 的 lowering
- fusion / decomposition
- layout planning
- memory planning / buffer reuse policy
- Triton lowering
- vendor runtime / device launch lowering

---

## 2. 当前项目状态

当前 `compiler-mlir` 已具备：

- `mini.constant`
- `mini.linear`
- `mini.relu`
- `mini.fused_linear_relu`
- `mini-canonicalize`
- `mini-const-fold`
- `mini-fusion`
- `mini-lower-to-linalg`
- `mini-cpu-lowering`

当前 CPU 主线已经可以做到：

- `mini.*`
- `linalg/tensor/arith`
- `bufferization/memref`
- `loops/cf`
- `llvm dialect`
- `llvm ir text`

也就是说，CPU 路线的“编译表示链路”已经打通，但还没有把“真正执行 / runtime / runner”收口成一个统一入口。

---

## 3. 推荐的 lowering 分层

## 3.1 Layer A: `mini` High-Level IR

这是项目自己的高层 IR 层，职责是表达小模型编译器的核心算子语义。

建议这一层长期保留的 op 类型：

- `mini.constant`
- `mini.linear`
- `mini.relu`
- `mini.add`
- `mini.mul`
- `mini.matmul`
- `mini.reshape`
- `mini.transpose`
- `mini.softmax`
- 后续可加 `mini.layernorm` / `mini.gelu`

这一层的目标：

- 让前端 bridge 过来的 IR 简洁可读
- 让图级优化 pass 更容易写
- 尽量保留模型语义，而不是一开始就丢进通用 op 海洋里

### 这一层适合做的 pass

- canonicalization
- constant fold
- DCE
- fusion
- decomposition 前准备

---

## 3.2 Layer B: Standard Tensor / Linalg Layer

这一层把 `mini` 自定义语义 lower 成 MLIR 标准计算语义。

核心 dialect：

- `arith`
- `tensor`
- `linalg`
- `func`

推荐 lowering：

- `mini.constant -> arith.constant`
- `mini.linear -> linalg.matmul + bias add`
- `mini.relu -> linalg.generic`
- `mini.add -> arith.addf` 或 `linalg.generic`
- `mini.mul -> arith.mulf` 或 `linalg.generic`
- `mini.reshape -> tensor.expand_shape / collapse_shape`
- `mini.transpose -> linalg.transpose`
- `mini.softmax -> 拆成标准 op 序列`

### 这一层适合做的 pass

- op decomposition
- canonicalize
- linalg-level fusion
- shape / type normalization
- tile-friendly rewrite

---

## 3.3 Layer C: Buffer / Memory Layer

这一层负责从 tensor 值语义进入内存语义。

核心 dialect：

- `bufferization`
- `memref`

典型 pass：

- `one-shot-bufferize`
- `convert-bufferization-to-memref`

### 这一层的职责

- 决定哪些 tensor 要落成 buffer
- 决定函数边界如何 bufferize
- 引入 `memref.alloc`
- 为后续 loop lowering 和 backend lowering 提供真实内存对象

### 后续可能需要自定义的方向

- memory planning
- buffer reuse
- inplace policy
- static / dynamic shape mixed strategy
- backend-specific memory space selection

---

## 3.4 Layer D: Schedule / Loop Layer

这一层负责把结构化算子转成显式计算过程。

核心 dialect：

- `scf`
- `cf`
- `affine`（可选）

典型 pass：

- `convert-linalg-to-loops`
- `convert-scf-to-cf`

### 这一层的职责

- 把整体 op 拆成循环
- 显式化遍历顺序
- 为 CPU / GPU mapping 提供调度结构

### 未来值得新增的自定义 pass

- tiling
- loop reorder
- fusion after tiling
- vectorization prep
- parallel mapping prep

---

## 3.5 Layer E: Backend-Specific Lowering

这一层才真正开始按后端分叉。

推荐分成四条主路线：

1. CPU
2. Triton
3. 通用 GPU dialect
4. vendor backend

---

## 4. CPU 路线

CPU 是当前最应该先做稳的主线。

## 4.1 推荐主链路

`mini`
-> `linalg/tensor/arith`
-> `bufferization/memref`
-> `loops/cf`
-> `llvm dialect`
-> `llvm ir`
-> runner / JIT / AOT

## 4.2 当前已落地

当前已有：

- `mini-lower-to-linalg`
- `mini-cpu-lowering`

`mini-cpu-lowering` 当前封装了：

- `mini-lower-to-linalg`
- `one-shot-bufferize`
- `drop-equivalent-buffer-results`
- `buffer-results-to-out-params`
- `convert-bufferization-to-memref`
- `convert-linalg-to-loops`
- `convert-scf-to-cf`
- `convert-cf-to-llvm`
- `convert-arith-to-llvm`
- `convert-index-to-llvm`
- `expand-realloc`
- `finalize-memref-to-llvm`
- `convert-func-to-llvm`
- `reconcile-unrealized-casts`

## 4.3 下一步建议

CPU 还应继续补：

- 统一 runner / execution harness
- LLVM IR 到执行的项目内脚本
- benchmark driver
- vector / SIMD 路线
- cache-aware tiling

---

## 5. Triton 路线

Triton 后端不适合一开始追求全覆盖。

推荐目标：

- 先只支持热点算子
- 再逐步支持 fused op

## 5.1 推荐支持的第一批热点

- `mini.linear`
- `mini.matmul`
- `mini.fused_linear_relu`

## 5.2 推荐 lowering 思路

`mini`
-> `linalg/tensor`
-> hotspot detection
-> Triton-friendly kernel plan
-> Triton kernel emission
-> launch/runtime bridge

## 5.3 需要新增的 pass

- `mini-triton-prepare`
  - 标记可下沉到 Triton 的 op
- `mini-triton-lowering`
  - 为热点 op 生成 Triton kernel plan
- `mini-triton-runtime-lowering`
  - 生成 host 侧 launch / parameter packing

## 5.4 关键工程点

- tile shape
- memory layout
- block/thread mapping
- kernel launch 参数
- 与 CPU reference backend 的数值对照

---

## 6. 通用 GPU dialect 路线

如果希望更贴近标准 MLIR GPU 生态，建议保留一条通用 GPU 路线。

## 6.1 推荐主链路

`mini`
-> `linalg/tensor`
-> tiling / mapping
-> `gpu` dialect
-> target-specific dialect
-> device binary / runtime launch

## 6.2 适用目标

- NVIDIA 路线：`gpu -> nvvm`
- AMD 路线：`gpu -> rocdl`

## 6.3 值得新增的 pass

- `mini-gpu-tile`
- `mini-gpu-map`
- `mini-gpu-launch-lowering`

## 6.4 这一条路线的意义

- 学习标准 MLIR GPU 编译主线
- 为 Triton 之外的 GPU 路线保留空间
- 为 vendor backend 抽象提供参考

---

## 7. 国内厂商芯片路线

对国内厂商芯片，不建议假设它们都适合直接走“CPU 路线”或“Triton 路线”。

建议分两类思考。

## 7.1 GPU-like 厂商芯片

这类设备通常具备：

- 类 CUDA / GPGPU 编程模型
- 类线程块 / 并行 kernel 模型
- LLVM-based 或 runtime-based codegen

推荐路线：

`mini`
-> `linalg/tensor`
-> tiling / mapping
-> vendor GPU dialect 或 runtime call layer
-> target backend

需要重点抽象：

- memory spaces
- launch model
- shared / local / global memory
- kernel ABI
- runtime API

## 7.2 NPU / 专用 AI 加速器

这类设备通常更像：

- graph executor
- op library
- command stream
- DMA + compute engine

推荐路线不是直接 lower 到 loops，而更像：

`mini`
-> `linalg/tensor`
-> partition
-> op legalization
-> memory/layout planning
-> vendor runtime graph / command IR

这类后端更重要的是：

- subgraph partition
- layout transform
- DMA scheduling
- runtime op emission

## 7.3 对当前项目的建议

当前阶段不要直接绑定某一家厂商。

应先在设计上预留：

- backend capability interface
- runtime bridge abstraction
- memory space abstraction
- launch / command abstraction

这样以后可以接：

- GPU-like vendor backend
- graph-runtime-style accelerator backend

---

## 8. 推荐新增的 pass / pipeline 列表

下面是建议在 `compiler-mlir` 里逐步新增的 pass 名字。

## 8.1 公共层

- `mini-canonicalize`
- `mini-const-fold`
- `mini-dce`
- `mini-fusion`
- `mini-lower-to-linalg`
- `mini-layout-lowering`
- `mini-tile-and-fuse`

## 8.2 CPU 层

- `mini-cpu-lowering`
- `mini-cpu-runner`（工具或脚本层）

## 8.3 Triton 层

- `mini-triton-prepare`
- `mini-triton-lowering`
- `mini-triton-runtime-lowering`

## 8.4 通用 GPU 层

- `mini-gpu-tile`
- `mini-gpu-map`
- `mini-gpu-lowering`

## 8.5 Vendor 层

- `mini-vendor-partition`
- `mini-vendor-layout`
- `mini-vendor-runtime-lowering`

---

## 9. 推荐的阶段顺序

建议按下面顺序推进：

### Phase 1

- 稳定 `mini -> linalg`
- 稳定 `mini-cpu-lowering`
- 增加 LLVM IR 翻译 / runner 文档

### Phase 2

- 增加更多 `mini` op
- 增加 linalg-level canonicalize / fusion / decomposition
- 增加 layout pass skeleton

### Phase 3

- 开始 Triton 热点路线
- 先支持 `linear` / `matmul` / `fused_linear_relu`

### Phase 4

- 增加通用 GPU dialect 路线
- 建立 thread/block mapping 思路

### Phase 5

- 设计 vendor backend abstraction
- 预留国产 GPU / NPU 路线接口

---

## 9.5 当前项目的 GPU 扩展实施方案

结合当前 `compiler-mlir` 已有实现，GPU 路线后续不建议“同时开很多方向”，而建议按下面顺序推进：

1. 融合相关 lowering
2. GPU 映射策略
3. host/device 边界与 memory space 处理
4. 算子选择 / 后端路线选择

原因是：

- 当前项目已经有 `mini -> linalg` 基础
- 已经有 `linear + relu -> fused_linear_relu` 的最小融合样例
- 已经有 `mini-gpu-lowering` 主链路
- 已经有 `mini-gpu-host-shared` 的 host/device 边界雏形

也就是说，继续做 GPU 路线时，最稳的做法不是“另起一套新架构”，而是顺着现有 pass/pipeline 逐层增强。

### 第一步：融合相关 lowering

这是最适合最先扩展的方向。

#### 目标

- 扩大高层融合覆盖面
- 为后续 GPU kernel 生成提供更稳定的高层语义单元
- 让 GPU 路线的 kernel 边界更清晰

#### 推荐先支持的模式

- `mini.linear + mini.relu`
  - 当前已支持，作为基线
- `mini.matmul + mini.add`
- `mini.matmul + mini.add + mini.relu`
- `mini.linear + bias-add`
- 后续再考虑：
  - `mini.softmax` 局部模式
  - `mini.layernorm` 局部模式

#### 在当前项目中的落点

- 继续扩展 `mini-canonicalize`
- 继续扩展 `mini-fusion`
- 必要时新增更明确的 pass，例如：
  - `mini-gpu-fusion`

#### 建议修改位置

- `projects/mini-ai-compiler/compiler-mlir/lib/Passes.cpp`
- `projects/mini-ai-compiler/compiler-mlir/include/MiniCompiler/Passes.h`

#### 这一阶段的产物

- 新的融合 pattern
- 新的测试样例
- GPU lowering 前后的 IR 对比

#### 为什么先做这个

因为它直接影响：

- 一个 kernel 对应哪些计算
- GPU route 后面看到的是“大一点的融合单元”还是“碎小 op”

这一步越稳，后面的 tile / map / runtime 设计越不容易反复推翻。

---

### 第二步：GPU 映射策略

这一步负责回答：

- 怎么切 tile
- 怎么映射 block / thread
- 哪些 loop 归 block
- 哪些 loop 归 thread

#### 目标

- 把当前“能 lower 到 `gpu.launch`”提升到“按可解释规则 lower 到 `gpu.launch`”
- 让 GPU 路线具备明确的 schedule 层

#### 推荐新增的 pass

- `mini-gpu-tile`
- `mini-gpu-map`

#### 推荐第一版策略

第一版不需要复杂 cost model，可以先做规则版：

- 小 elementwise：
  - 直接 map 到 thread-level 并行
- `matmul` / `linear`：
  - 固定 tile 大小
  - 固定 block/thread 映射
- `fused_linear_relu`：
  - 先沿用 `linear` 的 tile 方案
  - `relu` 作为 epilogue 留在同一 kernel 内

#### 在当前项目中的落点

当前 `mini-gpu-lowering` 已经包含：

- `convert-linalg-to-parallel-loops`
- `gpu-map-parallel-loops`
- `convert-parallel-loop-to-gpu`
- `gpu-kernel-outlining`

下一步建议做法是：

- 在这些官方 pass 之前插入自定义策略 pass
- 先显式写出 tile / map 规则
- 再让后续官方 pass 继续接管

#### 建议修改位置

- `projects/mini-ai-compiler/compiler-mlir/lib/Passes.cpp`
- `projects/mini-ai-compiler/compiler-mlir/LOWERING_ROADMAP.md`
- `projects/mini-ai-compiler/compiler-mlir/test/`

#### 这一阶段的产物

- 一个可解释的 `mini-gpu-tile` pass
- 一个可解释的 `mini-gpu-map` pass
- 至少一组固定 tile 参数的 demo
- tile size 可通过 pipeline option 显式覆盖，例如 `tile-sizes=4,2`

---

### 第三步：host/device 边界与 memory space 处理

这一步回答的是：

- 哪些 buffer 留在 host
- 哪些要转成 device 可见
- 哪些适合 `host_shared`
- kernel 参数怎么组织

#### 当前基础

项目里已经有：

- `mini-gpu-host-shared`

它已经在做：

- 把 `gpu.launch_func` 使用到的 host memref 改写成 `gpu.alloc host_shared`
- 为 GPU 可见 buffer 插入 `memref.copy`
- 对只读源只做 copy-in，不做 copy-back
- 对可写源保守地做 copy-in + copy-back

#### 下一步建议

把这部分从“能跑的样例逻辑”提升到“有策略的 memory pass”：

- 明确输入 / 输出 / 中间值的分类
- 明确哪些值只读，哪些值可写
- 明确哪些 copy 是必须的，哪些可省
- 后续为 shared memory / local memory / global memory 预留抽象

#### 推荐新增方向

- `mini-gpu-memory-plan`
- 或扩展现有 `mini-gpu-host-shared`

#### 推荐第一版能力

- 输入 buffer 分类
- 输出 buffer 分类
- 常量 tensor 的 device 可见策略
- 中间 memref 的最小复制策略
- kernel 参数顺序的固定规则
- 多次 launch 时按 launch 就近物化临时 shared buffer，避免跨 launch 复用陈旧副本

#### 为什么这一步排在映射之后

因为 tile / block / thread 方案确定后，才能更自然地讨论：

- 哪些数据要提前搬运
- 哪些数据值得放到更快的 memory space

#### 建议修改位置

- `projects/mini-ai-compiler/compiler-mlir/lib/Passes.cpp`
- `projects/mini-ai-compiler/compiler-mlir/tools/mini-compiler-gpu-runner.cpp`
- `projects/mini-ai-compiler/compiler-mlir/runtime/`

#### 这一阶段的产物

- 更清晰的 host/device buffer 规则
- 更稳定的 GPU runner 输入约定
- 更容易扩展到云端执行的 runtime 契约

---

### 第四步：算子选择 / 后端路线选择

这是 GPU 路线和 Triton / library backend 的分叉层。

它回答的是：

- 哪些 op 走通用 MLIR GPU 路线
- 哪些 op 走 Triton
- 哪些 op 未来应走 cuBLAS / CUTLASS 或其他 vendor library

#### 原则

第一版不要做复杂 cost model，先做规则选择即可。

#### 推荐第一版规则

- 通用 MLIR GPU：
  - `relu`
  - `add`
  - `mul`
- Triton 候选：
  - `fused_linear_relu`
  - 热点 elementwise fusion
- library 候选：
  - 大尺寸 `matmul`
  - 大尺寸 `linear`

#### 推荐实现方式

先做“标记与分流”，不要求一开始就完成真实后端接入：

- 给 op 打 attribute
- 或者新增一个中间 pass：
  - `mini-backend-select`
  - `mini-triton-prepare`

#### 为什么把它放最后

因为只有当前面三步比较稳定时，路线选择才不会频繁失效：

- 融合单元要先稳定
- tile/map 要先大致稳定
- memory / launch 约定要先稳定

否则“应该走哪条路线”的判断标准会不断变化。

#### 建议修改位置

- `projects/mini-ai-compiler/compiler-mlir/lib/Passes.cpp`
- `projects/mini-ai-compiler/compiler-mlir/include/MiniCompiler/Passes.h`
- 后续可能新增：
  - `projects/mini-ai-compiler/compiler-mlir/lib/Strategy.cpp`
  - `projects/mini-ai-compiler/compiler-mlir/include/MiniCompiler/Strategy.h`

#### 这一阶段的产物

- 一版规则式 strategy selection
- 通用 GPU / Triton / library 三类候选分流
- 为后续真实 Triton / cuBLAS / CUTLASS 接入预留稳定入口

---

### 小结：为什么按这个顺序

建议顺序是：

1. 融合
2. 映射
3. 边界与 memory
4. 路线选择

因为它们的依赖关系基本就是：

- 先知道“一个 kernel 单元是什么”
- 再知道“这个 kernel 怎么映射到 GPU”
- 再知道“它的数据怎么进出 GPU、怎么摆放”
- 最后才决定“这个单元是否根本不该走这条 GPU 路，而应转去 Triton 或库调用”

这是当前 `compiler-mlir` 最适合的推进方式。

---

## 10. 当前项目的最终建议

对 `compiler-mlir`，建议明确以下定位：

- **CPU**
  - 正式主线
  - 最先跑通完整闭环
- **Triton**
  - 热点高性能后端
  - 不要求全覆盖
- **通用 GPU dialect**
  - 标准 MLIR 工程主线参考
- **vendor backend abstraction**
  - 为国内厂商芯片保留扩展能力

一句话总结：

> `compiler-mlir` 不应该被做成“只有 CPU 的教学小工具”，而应该被推进成“公共 lowering + 多后端分叉”的正式 MLIR 编译器骨架。

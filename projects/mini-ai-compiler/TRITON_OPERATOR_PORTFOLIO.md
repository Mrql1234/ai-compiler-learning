# Triton 算子作品集实现文档

## 1. 文档目的

本文档用于把一套面向 `AI 编译器 / 编译后端 / kernel compiler` 岗位的 Triton 算子作品集正式落到项目文档中。

目标不是单纯“会跑一个 Triton demo”，而是形成一套可以在面试里系统展示的能力闭环：

- 能讲清常见热点算子的数学语义与 shape 规则
- 能用 `Triton` 实现核心 kernel
- 能做正确性验证与 benchmark
- 能把 kernel 设计与 `fusion / lowering / codegen / strategy selection` 串起来

本文档是一个“文档先行”的实现版本，后续代码可以按这里约定的目录、入口文件和命令逐步落地。

## 2. 作品集范围

第一版作品集只做 5 个核心算子，保持“小而强”：

1. `triton_matmul.py`
2. `triton_fused_linear_relu.py`
3. `triton_softmax.py`
4. `triton_layernorm.py`
5. `triton_flash_attention_simplified.py`

选择原因：

- `matmul` 是最核心的 AI 计算主干
- `fused linear + relu` 体现图融合与 epilogue 融合
- `softmax` 体现数值稳定与 reduction
- `layernorm` 体现归约、统计量计算与融合空间
- `flash attention simplified` 体现 block-wise 设计和大模型热点算子理解

## 3. 推荐目录布局

建议在 `projects/mini-ai-compiler/` 下新增如下目录：

```text
portfolio/
  triton_ops/
    __init__.py
    triton_matmul.py
    triton_fused_linear_relu.py
    triton_softmax.py
    triton_layernorm.py
    triton_flash_attention_simplified.py
  tests/
    __init__.py
    test_triton_matmul.py
    test_triton_fused_linear_relu.py
    test_triton_softmax.py
    test_triton_layernorm.py
    test_triton_flash_attention_simplified.py
  benchmarks/
    __init__.py
    bench_triton_matmul.py
    bench_triton_fused_linear_relu.py
    bench_triton_softmax.py
    bench_triton_layernorm.py
    bench_triton_flash_attention_simplified.py
```

这样做的好处是：

- 与现有 `tools/`、`tests/`、`benchmarks/` 风格一致
- 把“作品集训练代码”和主线教学编译器代码隔离开
- 后续更容易单独展示、迁移或抽成面试作品

## 4. 统一入口约定

### 4.1 建议入口文件

- `portfolio/triton_ops/triton_matmul.py`
- `portfolio/triton_ops/triton_fused_linear_relu.py`
- `portfolio/triton_ops/triton_softmax.py`
- `portfolio/triton_ops/triton_layernorm.py`
- `portfolio/triton_ops/triton_flash_attention_simplified.py`

### 4.2 建议运行命令

```bash
cd projects/mini-ai-compiler
python3 -m portfolio.triton_ops.triton_matmul
python3 -m portfolio.triton_ops.triton_fused_linear_relu
python3 -m portfolio.triton_ops.triton_softmax
python3 -m portfolio.triton_ops.triton_layernorm
python3 -m portfolio.triton_ops.triton_flash_attention_simplified
```

### 4.3 建议测试命令

```bash
cd projects/mini-ai-compiler
python3 -m portfolio.tests.test_triton_matmul
python3 -m portfolio.tests.test_triton_fused_linear_relu
python3 -m portfolio.tests.test_triton_softmax
python3 -m portfolio.tests.test_triton_layernorm
python3 -m portfolio.tests.test_triton_flash_attention_simplified
```

### 4.4 建议 benchmark 命令

```bash
cd projects/mini-ai-compiler
python3 -m portfolio.benchmarks.bench_triton_matmul
python3 -m portfolio.benchmarks.bench_triton_fused_linear_relu
python3 -m portfolio.benchmarks.bench_triton_softmax
python3 -m portfolio.benchmarks.bench_triton_layernorm
python3 -m portfolio.benchmarks.bench_triton_flash_attention_simplified
```

## 5. 通用实现模板

每个算子入口文件建议统一包含以下内容：

1. `PyTorch` 参考实现
2. `Triton kernel`
3. Python 包装函数
4. correctness 对拍入口
5. benchmark 入口
6. 命令行主函数

推荐统一主函数输出：

- 输入 shape
- 关键 tile 参数
- `torch` 与 `triton` 的最大误差
- 是否 `allclose`
- 简单 benchmark 结果

推荐统一测试内容：

- 小 shape 正确性
- 边界 shape
- 非整 tile shape
- 不同 dtype 的基础覆盖

## 6. 五个算子的实现要求

### 6.1 `triton_matmul.py`

#### 公式

`C = A x B`

其中：

- `A` shape 为 `[M, K]`
- `B` shape 为 `[K, N]`
- `C` shape 为 `[M, N]`

#### 目标能力

- 理解 `M / N / K` 三维分块
- 理解 `K` 维累加
- 理解寄存器累加与访存块加载
- 能解释 tile shape 为什么影响性能

#### Triton 实现重点

- 按二维 program grid 映射 `M x N` 输出 tile
- 沿 `K` 维循环加载 `A` 和 `B` 子块
- 用 accumulator 做局部累加
- 使用 mask 处理边界

#### 正确性验证

- 与 `torch.matmul` 对拍
- 覆盖方阵和非方阵
- 覆盖 `M/N/K` 不是 tile 整数倍的情况

#### benchmark 方法

- 固定 `dtype=float16` 或 `float32`
- 测试 `512x512x512`、`1024x1024x1024`
- 与 `torch.matmul` 做时间对比

#### 与编译器优化的关系

- `linear` 最终常常落到 `matmul` 或库 `GEMM`
- `matmul` 是后端策略选择的关键目标
- 编译器会决定它走 `library`、`generated kernel` 还是 `triton`

### 6.2 `triton_fused_linear_relu.py`

#### 公式

`Y = relu(X x W^T + b)`

其中：

- `X` shape 为 `[M, K]`
- `W` shape 为 `[N, K]`
- `b` shape 为 `[N]`
- `Y` shape 为 `[M, N]`

#### 目标能力

- 理解 `linear` 与 `matmul` 的关系
- 理解 bias epilogue
- 理解为什么 `relu` 融合到 epilogue 有收益

#### Triton 实现重点

- 复用 `matmul` 的 tile 结构
- 在写回前执行 `+ b` 与 `relu`
- 避免中间 tensor 写回显存

#### 正确性验证

- 与 `torch.nn.functional.linear` + `torch.relu` 对拍
- 覆盖带 bias 与不带 bias
- 覆盖不同 batch size

#### benchmark 方法

- 对比：
  - `torch.matmul + bias + relu`
  - fused Triton kernel
- 重点观测融合后是否减少 kernel launch 与显存访问

#### 与编译器优化的关系

- 这是典型的图融合题
- 在图层面可识别 `linear + relu`
- 在后端层面要为 fused op 选择专用 kernel 路径

### 6.3 `triton_softmax.py`

#### 公式

`softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`

通常按最后一维或按行做。

#### 目标能力

- 理解数值稳定为什么必须减 `max`
- 理解两次 reduction 结构
- 理解 row-wise softmax 的线程映射

#### Triton 实现重点

- 先做 row max reduction
- 再做 `exp(x - max)`
- 再做 row sum reduction
- 最后归一化写回

#### 正确性验证

- 与 `torch.softmax` 对拍
- 覆盖较大和较小输入值
- 重点检查数值稳定性

#### benchmark 方法

- 测试不同 `batch x hidden`
- 重点观察 reduction 对性能的影响
- 与 `torch.softmax` 做基线对比

#### 与编译器优化的关系

- attention 中常见 `scale + mask + softmax`
- 编译器经常尝试把这些前后处理融合
- softmax 是从图融合进入 kernel 融合的典型例子

### 6.4 `triton_layernorm.py`

#### 公式

对每一行输入 `x`：

`mean = sum(x) / N`

`var = sum((x - mean)^2) / N`

`y = (x - mean) / sqrt(var + eps) * gamma + beta`

#### 目标能力

- 理解 mean/variance 两类统计量
- 理解归一化与 affine 变换
- 理解 `eps` 的作用

#### Triton 实现重点

- 行级 reduction
- 使用同一 row tile 计算 mean 与 var
- 写回前完成 affine
- 处理 hidden dim 非整 tile 场景

#### 正确性验证

- 与 `torch.nn.functional.layer_norm` 对拍
- 覆盖不同 hidden size
- 覆盖 `gamma/beta` 存在与缺失场景

#### benchmark 方法

- 测试典型 transformer hidden size，如 `768 / 1024 / 4096`
- 比较 `torch` 实现与 Triton 实现

#### 与编译器优化的关系

- 常与 residual/add 融合
- 是 transformer block 中的重要热点
- 也是 strategy selection 中很适合专门下沉到 Triton 的算子

### 6.5 `triton_flash_attention_simplified.py`

#### 公式

基础 attention：

`S = Q x K^T`

`P = softmax(S)`

`O = P x V`

简化版 `flash attention` 的目标不是完整工业实现，而是理解 block-wise attention 与 online softmax 的核心思路。

#### 目标能力

- 理解普通 attention 的中间矩阵为什么大
- 理解为什么要做 block-wise 处理
- 理解 online softmax 的核心更新逻辑
- 能解释为什么这是编译器和算子开发共同关心的热点

#### Triton 实现重点

- 按 `Q` block 和 `K/V` block 分块
- 累积 block 级别的 `max` 与 `sum`
- 避免显式物化完整 `QK^T`
- 最后输出 `O`

#### 正确性验证

- 先与 naive attention 的 `torch` 实现对拍
- 限定小规模 shape 做调试
- 单独检查 mask 与数值稳定逻辑

#### benchmark 方法

- 第一版不追求极致性能
- 重点对比 naive attention 的访存思路
- benchmark 中强调“思路正确性”与“中间张量减少”

#### 与编译器优化的关系

- attention 是大模型编译器最核心的热点之一
- 编译器会尝试：
  - pattern 识别
  - fused attention 路线选择
  - 选择库、Triton 或自生成 kernel
- 这个作品最能体现“图编译 + kernel 设计”的结合能力

## 7. 推荐学习与落地顺序

推荐严格按下面顺序实现，不建议一开始先做 `flash attention`：

1. `triton_matmul.py`
2. `triton_fused_linear_relu.py`
3. `triton_softmax.py`
4. `triton_layernorm.py`
5. `triton_flash_attention_simplified.py`

原因：

- `matmul` 是后续大部分算子的基础
- `fused linear + relu` 是最容易讲清楚的融合题
- `softmax` 和 `layernorm` 让你掌握 reduction
- `flash attention` 最后做，成功率更高

## 8. 八周学习与实现规划

### 第 1 周

- 搭 Triton 环境
- 写最小 `matmul` 骨架
- 补 PyTorch 对拍

### 第 2 周

- 优化 `matmul` tile 参数
- 加 benchmark
- 整理 `matmul` 文档说明

### 第 3 周

- 实现 `fused_linear_relu`
- 补 bias 与 relu epilogue

### 第 4 周

- 实现 `softmax`
- 重点处理数值稳定

### 第 5 周

- 实现 `layernorm`
- 覆盖 `gamma / beta / eps`

### 第 6 周

- 阅读并整理 `flash attention` 思路
- 先写 naive attention baseline

### 第 7 周

- 实现简化版 `flash attention`
- 补 correctness 调试脚本

### 第 8 周

- 完成统一 README
- 补全部 tests 与 benchmarks
- 准备面试讲稿

## 9. 面试展示建议

每个算子都建议准备一页自己的讲稿，固定包含：

1. 算子公式与 shape
2. 为什么它重要
3. Triton kernel 怎么映射 program/grid
4. 为什么这样分块
5. 正确性怎么验证
6. 性能瓶颈在哪
7. 如果放进编译器，通常在哪一层做融合或选择后端

这样面试时就不只是“会写 kernel”，而是能体现：

- 懂图编译
- 懂后端
- 懂热点算子
- 懂工程验证

## 10. 完成标准

这套作品集完成时，至少应满足：

- 5 个算子入口都能单独运行
- 每个算子都有 `torch` 对拍
- 每个算子都有 benchmark
- 每个算子都能解释其 tile 设计
- 每个算子都能说明它与编译器优化的关系
- 有一份总 README 或展示文档可以直接给面试官看

## 11. 当前状态

截至当前版本：

- 本文档已落地
- `README.md` 已同步记录入口文件与运行命令约定
- 具体 `portfolio/` 代码目录尚未开始实现

因此，当前完成的是“文档版作品集实现”，下一步再进入代码实现阶段最合适。

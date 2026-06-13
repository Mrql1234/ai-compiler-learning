# Triton 算子开发与自动迭代优化 Agent 规格说明

本文档把当前已经落地的 Triton Agent 原型、目录约定、执行入口和后续扩展方向统一收敛到 `perf/` 目录下，便于后续直接在云主机上继续迭代。

## 1. 项目目标

这个 Agent 的目标不是“一次性生成一个能跑的 kernel”，而是围绕热点算子建立一条可持续迭代的闭环：

`结构化规格 -> 候选配置生成 -> correctness / benchmark -> profiling 诊断 -> 下一轮优化建议 -> 经验记忆`

当前优先支持的算子方向：

- `fused_linear_relu`
- `matmul`
- `softmax`
- `layernorm`

## 2. 当前落地范围

当前原型已经具备：

- 统一的结构化规格入口
- 统一的 Agent CLI 入口
- 候选配置生成
- dry-run 计划生成
- `fused_linear_relu` 的 Triton benchmark / profile 命令编排
- `ncu` 文本解析与瓶颈分类
- 下一轮实验建议生成
- 最小经验记忆落盘

当前分层状态：

- `fused_linear_relu`
  - 已接入可执行闭环
- `matmul`
  - 已接入规格、候选生成、经验记忆、Nsight 诊断
  - 仍是 planner-only
- `softmax`
  - 已接入规格、候选生成、经验记忆、Nsight 诊断
  - 仍是 planner-only
- `layernorm`
  - 已接入规格、候选生成、经验记忆、Nsight 诊断
  - 仍是 planner-only

## 3. 目录约定

核心入口文件：

- `projects/mini-ai-compiler/compiler-mlir/scripts/triton_operator_agent.py`
- `projects/mini-ai-compiler/compiler-mlir/scripts/triton_operator_agent_lib.py`

`perf/` 目录下与这个 Agent 直接相关的内容：

- `perf/specs/`
  - 结构化算子规格
- `perf/cases/`
  - benchmark case
- `perf/configs/`
  - Triton 配置与 sweep 空间
- `perf/notes/`
  - 迭代记录
- `perf/skills/ncu-analysis/`
  - vendored 的 Nsight Compute 分析 skill
- `perf/TRITON_OPERATOR_AGENT_SPEC.md`
  - 当前这份整合规格说明
- `perf/CODEX_SKILLS_MIGRATION.md`
  - 云主机迁移说明

运行产物默认落到：

- `perf/runs/agent_runs/<spec_name>/plan.json`
- `perf/runs/agent_runs/<spec_name>/summary.json`
- `perf/runs/agent_runs/<spec_name>/best_result.json`
- `perf/runs/agent_runs/<spec_name>/analysis.json`
- `perf/runs/agent_runs/<spec_name>/report.md`
- `perf/agent_memory/triton_operator_history.json`

## 4. 结构化规格合同

当前规格 JSON 统一包含：

- `name`
- `operation`
- `dtype`
- `problem`
- `layout`
- `hardware`
- `constraints`
- `budgets`
- `artifacts`
- `candidate_space`
- `goals`
- `notes`

当前样例：

- `perf/specs/triton_agent_fused_linear_relu_a10.json`
- `perf/specs/triton_agent_matmul_a10.json`
- `perf/specs/triton_agent_softmax_a10.json`
- `perf/specs/triton_agent_layernorm_a10.json`

## 5. Agent 模式

### 5.1 `plan`

职责：

- 解析规格
- 选择算子适配器
- 生成候选配置
- 输出实验计划

适合：

- 本机无 GPU
- 想先在云主机运行前检查候选空间是否合理

### 5.2 `tune`

职责：

- 执行候选
- 汇总 benchmark 结果
- 选择当前最优候选
- 在可用时触发 profiling 和 profile-guided resweep

当前真正接通可执行链路的算子：

- `fused_linear_relu`

### 5.3 `analyze`

职责：

- 读取已有的 `ncu --page details` 文本
- 分类瓶颈
- 生成下一轮实验建议

适合：

- benchmark 和 profile 已在云主机上跑完
- 想在 Agent 层统一沉淀诊断逻辑

## 6. 当前推荐命令

### 6.1 本机无 GPU 时先做计划

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode plan \
  --dry-run
```

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_matmul_a10.json \
  --mode plan \
  --dry-run
```

### 6.2 云主机有 GPU 时做闭环调优

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode tune \
  --max-candidates 8 \
  --max-iterations 2
```

如果只是先验证命令拼装：

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode tune \
  --dry-run \
  --max-candidates 4 \
  --max-iterations 1
```

### 6.3 已有 `ncu` 文本时直接做分析

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode analyze \
  --ncu-details /path/to/iter_best_ncu_details.txt \
  --run-dir perf/runs/agent_runs/analyze_linear_relu
```

## 7. 云主机推进顺序建议

推荐你在云主机上按这个顺序继续：

1. 先跑 `plan --dry-run`
2. 再跑 `tune --dry-run`
3. 确认依赖没问题后，跑 `fused_linear_relu` 的真实 `tune`
4. 把 `summary.json`、`analysis.json`、`report.md` 留档
5. 根据结论补 `matmul` 的独立 Triton benchmark
6. 再把 `matmul` 接入第二个可执行闭环

## 8. 下一步建议

最优先的后续开发项：

- 新增 `scripts/triton_matmul_bench.py`
- 为 `matmul` 增加 case/config
- 让 `matmul` 从 planner-only 升级为可执行闭环

次优先：

- 为 `softmax` 增加 row-wise Triton baseline
- 为 `layernorm` 增加 row-wise Triton baseline
- 把最终成熟配置反接到 `compiler-mlir` 的 backend selection

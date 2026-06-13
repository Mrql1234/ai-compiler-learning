# Triton Agent 规格目录

这个目录用于保存面向 `scripts/triton_operator_agent.py` 的结构化算子规格。

每个规格文件都统一描述：

- `operation`
- `dtype`
- `problem`
- `layout`
- `hardware`
- `constraints`
- `budgets`
- `candidate_space`

当前样例：

- `triton_agent_fused_linear_relu_a10.json`
  - 当前唯一接入可执行 Triton benchmark / profile 闭环的样例
- `triton_agent_matmul_a10.json`
  - `matmul` 的 planner-only 样例
- `triton_agent_softmax_a10.json`
  - `softmax` 的 planner-only 样例
- `triton_agent_layernorm_a10.json`
  - `layernorm` 的 planner-only 样例

主要入口文件：

- `projects/mini-ai-compiler/compiler-mlir/scripts/triton_operator_agent.py`
- `projects/mini-ai-compiler/compiler-mlir/scripts/triton_operator_agent_lib.py`

配套整合文档：

- `projects/mini-ai-compiler/compiler-mlir/perf/TRITON_OPERATOR_AGENT_SPEC.md`
- `projects/mini-ai-compiler/compiler-mlir/perf/CODEX_SKILLS_MIGRATION.md`

常用命令：

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
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode tune \
  --dry-run \
  --max-candidates 4 \
  --max-iterations 1
```

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_matmul_a10.json \
  --mode plan \
  --dry-run
```

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode analyze \
  --ncu-details /path/to/iter_best_ncu_details.txt \
  --run-dir perf/runs/agent_runs/analyze_linear_relu
```

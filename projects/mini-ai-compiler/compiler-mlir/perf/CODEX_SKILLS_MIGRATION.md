# 云主机上的 Codex Skills 迁移说明

本文档说明如何把当前本机 Codex 里的迭代优化 skill 迁移到云主机，重点覆盖这个 Triton Agent 最直接依赖的 `ncu-analysis`。

## 1. 先说结论

最稳妥的做法不是依赖本机 `C:\\Users\\QL\\.codex\\skills\\...` 目录，而是把关键 skill 直接 vendoring 到仓库里。

当前仓库已经内置：

- `projects/mini-ai-compiler/compiler-mlir/perf/skills/ncu-analysis/`

所以你在云主机上只要同步这份仓库，就已经拿到了 skill 本体。

## 2. skill 目录内容

当前 vendored 的 `ncu-analysis` 包含：

- `SKILL.md`
- `references/metrics-guide.md`
- `references/triton-tuning-map.md`
- `scripts/extract_ncu_key_lines.py`
- `agents/openai.yaml`

它的用途是：

- 读 `ncu --page details` 文本
- 抽取高信号行
- 做 bottleneck 分类
- 生成下一轮 Triton 调参建议

## 3. 迁移方式

### 方式 A：云主机上也跑 Codex

如果云主机上也是 Codex 环境，推荐直接把 vendored skill 安装到云主机用户目录下的 `~/.codex/skills/`。

仓库内已经提供安装脚本：

- `projects/mini-ai-compiler/compiler-mlir/perf/skills/install_ncu_analysis_skill.sh`

用法：

```bash
cd projects/mini-ai-compiler/compiler-mlir

./perf/skills/install_ncu_analysis_skill.sh
```

默认会复制到：

- `~/.codex/skills/ncu-analysis`

如果要装到自定义位置：

```bash
cd projects/mini-ai-compiler/compiler-mlir

./perf/skills/install_ncu_analysis_skill.sh /path/to/custom/codex-home/skills
```

安装完成后，你在云主机的 Codex 会话里就可以继续按 skill 方式使用它。

### 方式 B：云主机上不跑 Codex，只跑脚本和文档

如果云主机只是普通 shell / tmux / ssh 环境，不跑 Codex，也没关系。

这时直接复用仓库内脚本和文档即可：

- 技能说明：
  - `perf/skills/ncu-analysis/SKILL.md`
- 指标解释：
  - `perf/skills/ncu-analysis/references/metrics-guide.md`
- 参数建议映射：
  - `perf/skills/ncu-analysis/references/triton-tuning-map.md`
- 文本提取脚本：
  - `perf/skills/ncu-analysis/scripts/extract_ncu_key_lines.py`

示例：

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./perf/skills/ncu-analysis/scripts/extract_ncu_key_lines.py \
  /path/to/iter_best_ncu_details.txt
```

然后再结合：

```bash
cd projects/mini-ai-compiler/compiler-mlir

python3 ./scripts/triton_operator_agent.py \
  --spec perf/specs/triton_agent_fused_linear_relu_a10.json \
  --mode analyze \
  --ncu-details /path/to/iter_best_ncu_details.txt
```

## 4. 推荐的云主机工作流

如果你要在云主机上继续做这个项目，推荐顺序是：

1. 同步整个仓库
2. 安装 Python / Triton / PyTorch / Nsight 依赖
3. 先跑 `triton_operator_agent.py --mode plan --dry-run`
4. 再跑 `triton_operator_agent.py --mode tune`
5. 采集 `ncu` 文本
6. 用 vendored 的 `ncu-analysis` 做分析
7. 把结论沉淀回 `perf/runs/agent_runs/` 和 `perf/notes/`

## 5. 为什么要 vendoring 到仓库

这样做有几个直接好处：

- 云主机不依赖本机 Codex 安装目录
- skill 版本和项目代码可以一起提交、一起回滚
- 不管是不是 Codex 环境，都能直接用脚本和文档
- 后面如果你想加 `matmul` / `softmax` / `layernorm` 的专用调优说明，也可以继续按同样方式放进仓库

## 6. 后续建议

如果后面你发现还需要更多 cloud-only skill，建议也按同样方式放进仓库，比如：

- `perf/skills/triton-matmul-tuning/`
- `perf/skills/layernorm-diagnosis/`
- `perf/skills/softmax-profile-review/`

这样这个 Agent 项目的“优化知识库”就会逐步从 Codex 私有环境，迁移成仓库内可复用资产。

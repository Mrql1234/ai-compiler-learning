#!/usr/bin/env python3
"""Shared building blocks for the Triton operator development agent prototype."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_perf_common import (
    config_tag,
    ensure_directory,
    iter_sweep_configs,
    load_config_payload,
    quote_command,
    resolve_input_path,
    resolve_output_path,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_MEMORY = PROJECT_DIR / "perf" / "agent_memory" / "triton_operator_history.json"
SUPPORTED_OPERATIONS = {"fused_linear_relu", "matmul", "softmax", "layernorm"}
CONFIG_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages")


@dataclass(frozen=True)
class SearchBudget:
    max_candidates: int = 8
    max_iterations: int = 2
    warmup: int = 10
    repeat: int = 50
    profile_top_k: int = 1


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    operation: str
    dtype: str
    problem: dict[str, int]
    layout: dict[str, Any]
    hardware: dict[str, Any]
    constraints: dict[str, Any]
    budgets: SearchBudget
    artifacts: dict[str, Any]
    candidate_space: dict[str, list[int]]
    goals: dict[str, Any]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Candidate:
    operation: str
    config: dict[str, int]
    strategy: str
    reason: str
    implementation: str

    @property
    def tag(self) -> str:
        if {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages"} <= self.config.keys():
            return config_tag(self.config)
        parts = [self.strategy]
        for key, value in self.config.items():
            parts.append(f"{key.lower()}_{value}")
        return "_".join(parts)


@dataclass(frozen=True)
class AgentRunOptions:
    mode: str
    dry_run: bool
    emit_nvtx: bool
    device_index: int
    run_dir: Path
    memory_path: Path
    max_candidates: int | None = None
    max_iterations: int | None = None
    warmup: int | None = None
    repeat: int | None = None
    ncu_details: Path | None = None


def load_operator_spec(path: str | Path) -> OperatorSpec:
    payload = _load_json(path)
    operation = str(payload.get("operation", "")).strip()
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"unsupported operation '{operation}'; expected one of {sorted(SUPPORTED_OPERATIONS)}"
        )
    problem = payload.get("problem") or payload.get("shape")
    if not isinstance(problem, dict) or not problem:
        raise ValueError("operator spec must provide a non-empty 'problem' object")
    budgets_payload = payload.get("budgets", {})
    if not isinstance(budgets_payload, dict):
        raise ValueError("operator spec field 'budgets' must be an object")
    candidate_space = payload.get("candidate_space", {})
    if not isinstance(candidate_space, dict):
        raise ValueError("operator spec field 'candidate_space' must be an object")
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError("operator spec field 'notes' must be an array")
    return OperatorSpec(
        name=str(payload.get("name", "unnamed_triton_operator_spec")),
        operation=operation,
        dtype=str(payload.get("dtype", "float32")),
        problem={str(key): int(value) for key, value in problem.items()},
        layout=dict(payload.get("layout", {})),
        hardware=dict(payload.get("hardware", {})),
        constraints=dict(payload.get("constraints", {})),
        budgets=SearchBudget(
            max_candidates=int(budgets_payload.get("max_candidates", 8)),
            max_iterations=int(budgets_payload.get("max_iterations", 2)),
            warmup=int(budgets_payload.get("warmup", 10)),
            repeat=int(budgets_payload.get("repeat", 50)),
            profile_top_k=int(budgets_payload.get("profile_top_k", 1)),
        ),
        artifacts=dict(payload.get("artifacts", {})),
        candidate_space={
            str(key): [int(item) for item in value]
            for key, value in candidate_space.items()
            if isinstance(value, list)
        },
        goals=dict(payload.get("goals", {})),
        notes=[str(item) for item in notes],
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_input_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class OperatorAdapter:
    operation: str
    supports_execution: bool = False
    supports_profiling: bool = False

    def default_case_path(self, spec: OperatorSpec) -> Path | None:
        value = spec.artifacts.get("case")
        if value:
            return resolve_input_path(value)
        return None

    def default_config_path(self, spec: OperatorSpec) -> Path | None:
        value = spec.artifacts.get("config")
        if value:
            return resolve_input_path(value)
        return None

    def load_config_payload(self, spec: OperatorSpec) -> dict[str, Any] | None:
        config_path = self.default_config_path(spec)
        if config_path is None:
            return None
        return load_config_payload(config_path)

    def candidate_space(self, spec: OperatorSpec, config_payload: dict[str, Any] | None) -> dict[str, list[int]]:
        if spec.candidate_space:
            return {key: sorted(set(values)) for key, values in spec.candidate_space.items()}
        if config_payload and isinstance(config_payload.get("sweep"), dict):
            sweep = config_payload["sweep"]
            return {
                key: sorted(set(int(item) for item in values))
                for key, values in sweep.items()
                if isinstance(values, list) and values
            }
        return {}

    def build_initial_candidates(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None, limit: int
    ) -> list[Candidate]:
        raise NotImplementedError

    def build_iteration_candidates(
        self,
        spec: OperatorSpec,
        base_candidate: Candidate,
        diagnosis: dict[str, Any],
        config_payload: dict[str, Any] | None,
        limit: int,
    ) -> list[Candidate]:
        return []

    def benchmark_command(
        self,
        spec: OperatorSpec,
        candidate: Candidate,
        options: AgentRunOptions,
        json_output: Path,
    ) -> list[str]:
        raise NotImplementedError

    def profile_command(
        self,
        spec: OperatorSpec,
        candidate: Candidate,
        options: AgentRunOptions,
        out_dir: Path,
        tag: str,
    ) -> list[str]:
        raise NotImplementedError


class FusedLinearReluAdapter(OperatorAdapter):
    operation = "fused_linear_relu"
    supports_execution = True
    supports_profiling = True

    def default_case_path(self, spec: OperatorSpec) -> Path | None:
        path = super().default_case_path(spec)
        if path is not None:
            return path
        return PROJECT_DIR / "perf" / "cases" / "triton_linear_relu_f32_m1024_n1024_k1024.json"

    def default_config_path(self, spec: OperatorSpec) -> Path | None:
        path = super().default_config_path(spec)
        if path is not None:
            return path
        return PROJECT_DIR / "perf" / "configs" / "triton_linear_relu_a10.json"

    def build_initial_candidates(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None, limit: int
    ) -> list[Candidate]:
        configs: list[dict[str, int]]
        if config_payload is not None:
            configs = iter_sweep_configs(config_payload)
        else:
            configs = _default_matmul_like_configs(self.candidate_space(spec, config_payload))
        ranked = sorted(configs, key=lambda item: _matmul_like_priority(item, spec.problem))
        return [
            Candidate(
                operation=spec.operation,
                config=config,
                strategy="triton_gemm_fusion",
                reason="初始候选来自显式 sweep 空间，优先覆盖 tile、warp、stage 组合。",
                implementation="scripts/triton_linear_relu_bench.py",
            )
            for config in ranked[:limit]
        ]

    def build_iteration_candidates(
        self,
        spec: OperatorSpec,
        base_candidate: Candidate,
        diagnosis: dict[str, Any],
        config_payload: dict[str, Any] | None,
        limit: int,
    ) -> list[Candidate]:
        allowed = self.candidate_space(spec, config_payload)
        if not allowed:
            allowed = {
                "BLOCK_M": [64, 128],
                "BLOCK_N": [64, 128],
                "BLOCK_K": [32, 64],
                "GROUP_M": [4, 8],
                "num_warps": [2, 4, 8],
                "num_stages": [2, 3, 4],
            }
        base = dict(base_candidate.config)
        bottleneck = str(diagnosis.get("bottleneck", "mixed / inconclusive"))
        proposals: list[dict[str, int]] = []
        if bottleneck == "resource-limited":
            proposals.append(_step_config(base, allowed, "num_warps", -1))
            proposals.append(_step_config(base, allowed, "BLOCK_M", -1))
            proposals.append(_step_config(base, allowed, "BLOCK_N", -1))
        elif bottleneck == "memory-bound":
            proposals.append(_step_config(base, allowed, "GROUP_M", -1))
            proposals.append(_step_config(base, allowed, "BLOCK_K", +1))
            proposals.append(_step_config(base, allowed, "BLOCK_N", +1))
        elif bottleneck == "latency-hiding-limited":
            proposals.append(_step_config(base, allowed, "num_stages", +1))
            proposals.append(_step_config(base, allowed, "num_warps", +1))
            proposals.append(_step_config(base, allowed, "BLOCK_K", +1))
        else:
            proposals.append(_step_config(base, allowed, "GROUP_M", +1))
            proposals.append(_step_config(base, allowed, "BLOCK_M", +1))
            proposals.append(_step_config(base, allowed, "BLOCK_N", +1))
        unique: list[Candidate] = []
        seen: set[str] = set()
        for config in proposals:
            tag = config_tag(config)
            if tag in seen:
                continue
            seen.add(tag)
            unique.append(
                Candidate(
                    operation=spec.operation,
                    config=config,
                    strategy="profile_guided_resweep",
                    reason=f"根据 {bottleneck} 诊断结果，在 best config 周围做小步调整。",
                    implementation="scripts/triton_linear_relu_bench.py",
                )
            )
            if len(unique) >= limit:
                break
        return unique

    def benchmark_command(
        self,
        spec: OperatorSpec,
        candidate: Candidate,
        options: AgentRunOptions,
        json_output: Path,
    ) -> list[str]:
        case_path = self.default_case_path(spec)
        config_path = self.default_config_path(spec)
        if case_path is None or config_path is None:
            raise ValueError("fused_linear_relu benchmark requires case/config paths")
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "triton_linear_relu_bench.py"),
            "--case",
            str(case_path),
            "--config",
            str(config_path),
            "--config-source",
            "default",
            "--warmup",
            str(_effective_warmup(spec, options)),
            "--repeat",
            str(_effective_repeat(spec, options)),
            "--device-index",
            str(options.device_index),
            "--json-output",
            str(json_output),
        ]
        if options.emit_nvtx:
            command.append("--emit-nvtx")
        for key in CONFIG_KEYS:
            value = candidate.config.get(key)
            if value is None:
                continue
            cli_name = "--" + (key.replace("_", "-") if key.startswith("num_") else key)
            command.extend([cli_name, str(value)])
        return command

    def profile_command(
        self,
        spec: OperatorSpec,
        candidate: Candidate,
        options: AgentRunOptions,
        out_dir: Path,
        tag: str,
    ) -> list[str]:
        case_path = self.default_case_path(spec)
        config_path = self.default_config_path(spec)
        if case_path is None or config_path is None:
            raise ValueError("fused_linear_relu profile requires case/config paths")
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "triton_profile_iter.py"),
            "--case",
            str(case_path),
            "--config",
            str(config_path),
            "--config-source",
            "default",
            "--warmup",
            str(max(1, _effective_warmup(spec, options) // 5 or 1)),
            "--repeat",
            str(max(2, min(5, _effective_repeat(spec, options)))),
            "--device-index",
            str(options.device_index),
            "--tag",
            tag,
            "--out",
            str(out_dir),
        ]
        if options.emit_nvtx:
            command.append("--emit-nvtx")
        for key in CONFIG_KEYS:
            value = candidate.config.get(key)
            if value is None:
                continue
            cli_name = "--" + (key.replace("_", "-") if key.startswith("num_") else key)
            command.extend([cli_name, str(value)])
        return command


class PlannerOnlyAdapter(OperatorAdapter):
    def build_initial_candidates(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None, limit: int
    ) -> list[Candidate]:
        configs = self._planner_configs(spec, config_payload)
        return configs[:limit]

    def _planner_configs(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None
    ) -> list[Candidate]:
        raise NotImplementedError


class MatmulAdapter(PlannerOnlyAdapter):
    operation = "matmul"

    def _planner_configs(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None
    ) -> list[Candidate]:
        space = self.candidate_space(spec, config_payload)
        configs = _default_matmul_like_configs(space)
        configs = sorted(configs, key=lambda item: _matmul_like_priority(item, spec.problem))
        return [
            Candidate(
                operation=spec.operation,
                config=config,
                strategy="triton_blocked_matmul",
                reason="面向 GEMM 类算子，优先搜索 tile 复用、流水深度和 CTA 映射。",
                implementation="planner_only",
            )
            for config in configs
        ]


class SoftmaxAdapter(PlannerOnlyAdapter):
    operation = "softmax"

    def _planner_configs(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None
    ) -> list[Candidate]:
        columns = int(spec.problem.get("cols", spec.problem.get("n", 0)))
        space = self.candidate_space(spec, config_payload)
        block_sizes = space.get("BLOCK_SIZE", [128, 256, 512, 1024])
        warps = space.get("num_warps", [2, 4, 8])
        rows = space.get("ROWS_PER_PROGRAM", [1, 2, 4])
        configs: list[Candidate] = []
        for block_size in block_sizes:
            if block_size < min(128, max(1, columns)):
                continue
            for num_warps in warps:
                for rows_per_program in rows:
                    configs.append(
                        Candidate(
                            operation=spec.operation,
                            config={
                                "BLOCK_SIZE": int(block_size),
                                "num_warps": int(num_warps),
                                "ROWS_PER_PROGRAM": int(rows_per_program),
                            },
                            strategy="triton_online_softmax",
                            reason="Softmax 倾向优先让一行留在片上，减少中间写回与额外同步。",
                            implementation="planner_only",
                        )
                    )
        return sorted(configs, key=lambda item: abs(item.config["BLOCK_SIZE"] - max(columns, 128)))


class LayerNormAdapter(PlannerOnlyAdapter):
    operation = "layernorm"

    def _planner_configs(
        self, spec: OperatorSpec, config_payload: dict[str, Any] | None
    ) -> list[Candidate]:
        hidden = int(spec.problem.get("hidden", spec.problem.get("n", 0)))
        space = self.candidate_space(spec, config_payload)
        block_sizes = space.get("BLOCK_SIZE", _layernorm_block_candidates(hidden))
        warps = space.get("num_warps", [2, 4, 8])
        persistent = space.get("ROWS_PER_PROGRAM", [1, 2])
        configs: list[Candidate] = []
        for block_size in block_sizes:
            for num_warps in warps:
                for rows_per_program in persistent:
                    configs.append(
                        Candidate(
                            operation=spec.operation,
                            config={
                                "BLOCK_SIZE": int(block_size),
                                "num_warps": int(num_warps),
                                "ROWS_PER_PROGRAM": int(rows_per_program),
                            },
                            strategy="triton_rowwise_layernorm",
                            reason="LayerNorm 倾向用 row-wise reduction，并围绕 hidden size 选块大小。",
                            implementation="planner_only",
                        )
                    )
        return sorted(configs, key=lambda item: abs(item.config["BLOCK_SIZE"] - _next_power_of_2(max(hidden, 1))))


ADAPTERS: dict[str, OperatorAdapter] = {
    "fused_linear_relu": FusedLinearReluAdapter(),
    "matmul": MatmulAdapter(),
    "softmax": SoftmaxAdapter(),
    "layernorm": LayerNormAdapter(),
}


def run_agent(spec: OperatorSpec, options: AgentRunOptions) -> dict[str, Any]:
    adapter = ADAPTERS[spec.operation]
    run_dir = ensure_directory(options.run_dir)
    config_payload = adapter.load_config_payload(spec)
    plan_limit = options.max_candidates or spec.budgets.max_candidates
    max_iterations = options.max_iterations or spec.budgets.max_iterations
    initial_candidates = adapter.build_initial_candidates(spec, config_payload, plan_limit)
    plan_payload = {
        "spec": operator_spec_to_dict(spec),
        "run": _run_metadata(options),
        "adapter": {
            "operation": adapter.operation,
            "supports_execution": adapter.supports_execution,
            "supports_profiling": adapter.supports_profiling,
        },
        "initial_candidates": [candidate_to_dict(item) for item in initial_candidates],
    }
    write_json(run_dir / "plan.json", plan_payload)

    all_iterations: list[dict[str, Any]] = []
    seen_tags = {item.tag for item in initial_candidates}
    pending = initial_candidates
    best_result: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    for iteration_index in range(max_iterations):
        if not pending:
            break
        iteration_dir = ensure_directory(run_dir / f"iter_{iteration_index:02d}")
        iteration_payload = _evaluate_candidates(
            spec=spec,
            adapter=adapter,
            candidates=pending,
            options=options,
            iteration_dir=iteration_dir,
            iteration_index=iteration_index,
        )
        all_iterations.append(iteration_payload)
        best_in_iteration = iteration_payload.get("best_result")
        if best_in_iteration and _is_better_result(best_in_iteration, best_result):
            best_result = best_in_iteration
        if options.mode != "tune" or best_in_iteration is None:
            continue
        if not adapter.supports_profiling:
            break
        if iteration_index + 1 >= max_iterations:
            break
        diagnosis = _maybe_profile_and_diagnose(
            spec=spec,
            adapter=adapter,
            best_result=best_in_iteration,
            options=options,
            iteration_dir=iteration_dir,
            config_payload=config_payload,
        )
        if not diagnosis:
            break
        next_candidates = adapter.build_iteration_candidates(
            spec=spec,
            base_candidate=_candidate_from_result(best_in_iteration),
            diagnosis=diagnosis,
            config_payload=config_payload,
            limit=plan_limit,
        )
        pending = [item for item in next_candidates if item.tag not in seen_tags]
        for item in pending:
            seen_tags.add(item.tag)
        if not pending:
            break

    if diagnosis is None and options.ncu_details is not None:
        diagnosis = analyze_ncu_report(
            spec=spec,
            ncu_details_path=options.ncu_details,
            best_config=best_result.get("config") if best_result else None,
        )

    summary = {
        "spec": operator_spec_to_dict(spec),
        "run": _run_metadata(options),
        "iterations": all_iterations,
        "best_result": best_result,
        "diagnosis": diagnosis,
    }
    write_json(run_dir / "summary.json", summary)
    if best_result is not None:
        write_json(run_dir / "best_result.json", best_result)
    if diagnosis is not None:
        write_json(run_dir / "analysis.json", diagnosis)
    _write_report(run_dir, spec, summary)
    _update_memory(options.memory_path, spec, summary)
    return summary


def analyze_ncu_report(
    spec: OperatorSpec,
    ncu_details_path: str | Path,
    best_config: dict[str, int] | None = None,
) -> dict[str, Any]:
    text = resolve_input_path(ncu_details_path).read_text(encoding="utf-8")
    metrics = parse_ncu_details_text(text)
    bottleneck = classify_bottleneck(metrics)
    experiments = suggest_next_experiments(spec.operation, metrics, best_config or {})
    return {
        "operation": spec.operation,
        "spec_name": spec.name,
        "bottleneck": bottleneck,
        "metrics": metrics,
        "next_experiments": experiments,
    }


def parse_ncu_details_text(text: str) -> dict[str, float]:
    return {
        "achieved_occupancy_pct": _extract_float(text, r"Achieved Occupancy\s+%\s+([0-9.]+)"),
        "compute_sm_throughput_pct": _extract_float(
            text, r"Compute \(SM\) Throughput\s+%\s+([0-9.]+)"
        ),
        "memory_throughput_gbps": _extract_float(text, r"Memory Throughput\s+Gbyte/s\s+([0-9.]+)"),
        "l2_hit_rate_pct": _extract_float(text, r"L2 Hit Rate\s+%\s+([0-9.]+)"),
        "no_eligible_pct": _extract_float(text, r"No Eligible\s+%\s+([0-9.]+)"),
        "eligible_warps_per_scheduler": _extract_float(
            text, r"Eligible Warps Per Scheduler\s+warp\s+([0-9.]+)"
        ),
        "registers_per_thread": _extract_float(
            text, r"Registers Per Thread\s+register/thread\s+([0-9.]+)"
        ),
        "dynamic_shared_memory_kib": _extract_float(
            text, r"Dynamic Shared Memory Per Block\s+Kbyte/block\s+([0-9.]+)"
        ),
        "shared_bank_conflicts": _extract_float(text, r"results in ([0-9.]+) bank conflicts"),
        "shared_bank_conflict_wavefront_pct": _extract_float(
            text, r"which represent\s+([0-9.]+)% of the overall"
        ),
        "short_scoreboard_cycles": _extract_float(
            text,
            r"spends ([0-9.]+) cycles being stalled waiting for a scoreboard dependency",
        ),
    }


def classify_bottleneck(metrics: dict[str, float]) -> str:
    occupancy = metrics.get("achieved_occupancy_pct") or 0.0
    registers = metrics.get("registers_per_thread") or 0.0
    compute = metrics.get("compute_sm_throughput_pct") or 0.0
    dram = metrics.get("memory_throughput_gbps") or 0.0
    eligible = metrics.get("eligible_warps_per_scheduler") or 0.0
    no_eligible = metrics.get("no_eligible_pct") or 0.0
    bank_conflicts = metrics.get("shared_bank_conflicts") or 0.0
    l2_hit_rate = metrics.get("l2_hit_rate_pct") or 0.0
    if bank_conflicts > 0.0:
        return "resource-limited"
    if occupancy < 35.0 and registers >= 96.0:
        return "resource-limited"
    if eligible < 1.0 and no_eligible >= 25.0:
        return "latency-hiding-limited"
    if compute < 55.0 and dram >= 250.0 and l2_hit_rate < 75.0:
        return "memory-bound"
    if compute >= 65.0 and occupancy >= 40.0:
        return "compute-bound"
    if occupancy < 25.0:
        return "under-filled kernel"
    return "mixed / inconclusive"


def suggest_next_experiments(
    operation: str, metrics: dict[str, float], best_config: dict[str, int]
) -> list[dict[str, Any]]:
    bottleneck = classify_bottleneck(metrics)
    suggestions: list[dict[str, Any]] = []
    if bottleneck == "resource-limited":
        suggestions.append(
            {
                "reason": "寄存器或共享内存压力偏高，先尝试缩 tile 或减少并行宽度。",
                "changes": _filtered_changes(
                    best_config,
                    {
                        "num_warps": _maybe_decrement(best_config.get("num_warps")),
                        "BLOCK_M": _halve_if_possible(best_config.get("BLOCK_M")),
                    },
                ),
            }
        )
        suggestions.append(
            {
                "reason": "验证是不是 pipeline 深度把 occupancy 压低了。",
                "changes": _filtered_changes(
                    best_config,
                    {"num_stages": _maybe_decrement(best_config.get("num_stages"))},
                ),
            }
        )
    elif bottleneck == "memory-bound":
        suggestions.append(
            {
                "reason": "优先改善数据复用和 program ordering，看看 L2 / DRAM 是否回暖。",
                "changes": _filtered_changes(
                    best_config,
                    {
                        "GROUP_M": _toggle_group_m(best_config.get("GROUP_M")),
                        "BLOCK_K": _double_if_possible(best_config.get("BLOCK_K")),
                    },
                ),
            }
        )
        suggestions.append(
            {
                "reason": "如果是 row-major 访问，尝试更大的 N 方向 tile 提高写回连续性。",
                "changes": _filtered_changes(
                    best_config,
                    {"BLOCK_N": _double_if_possible(best_config.get("BLOCK_N"))},
                ),
            }
        )
    elif bottleneck == "latency-hiding-limited":
        suggestions.append(
            {
                "reason": "先验证更深 pipeline 能否提升隐藏访存延迟的能力。",
                "changes": _filtered_changes(
                    best_config,
                    {"num_stages": _maybe_increment(best_config.get("num_stages"))},
                ),
            }
        )
        suggestions.append(
            {
                "reason": "如果寄存器还扛得住，再增加 warp 数测试更多并发。",
                "changes": _filtered_changes(
                    best_config,
                    {"num_warps": _maybe_increment(best_config.get("num_warps"))},
                ),
            }
        )
    elif bottleneck == "compute-bound":
        suggestions.append(
            {
                "reason": "已经比较吃算力，下一步主要验证更大 tile 是否还能提升 tensor core / FMA 利用率。",
                "changes": _filtered_changes(
                    best_config,
                    {
                        "BLOCK_M": _double_if_possible(best_config.get("BLOCK_M")),
                        "BLOCK_N": _double_if_possible(best_config.get("BLOCK_N")),
                    },
                ),
            }
        )
    else:
        suggestions.append(
            {
                "reason": "先做一轮小步 resweep，把最有可能影响访存和 occupancy 的参数分开验证。",
                "changes": _filtered_changes(
                    best_config,
                    {
                        "GROUP_M": _toggle_group_m(best_config.get("GROUP_M")),
                        "num_stages": _maybe_increment(best_config.get("num_stages")),
                    },
                ),
            }
        )
    for index, item in enumerate(suggestions, start=1):
        item["id"] = f"{operation}_exp_{index}"
    return suggestions[:3]


def operator_spec_to_dict(spec: OperatorSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["budgets"] = asdict(spec.budgets)
    return payload


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return asdict(candidate) | {"tag": candidate.tag}


def _evaluate_candidates(
    spec: OperatorSpec,
    adapter: OperatorAdapter,
    candidates: list[Candidate],
    options: AgentRunOptions,
    iteration_dir: Path,
    iteration_index: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    for candidate_index, candidate in enumerate(candidates):
        json_output = iteration_dir / f"{candidate_index:03d}_{candidate.tag}.json"
        record = {
            "iteration": iteration_index,
            "candidate_index": candidate_index,
            "candidate": candidate_to_dict(candidate),
        }
        if options.mode == "plan":
            record["status"] = "planned"
        elif not adapter.supports_execution:
            record["status"] = "planner_only"
            record["message"] = "该算子当前只有规格化规划入口，尚未接入可执行 Triton benchmark。"
        else:
            command = adapter.benchmark_command(spec, candidate, options, json_output)
            record["command"] = command
            record["command_text"] = quote_command(command)
            if options.dry_run:
                record["status"] = "dry_run"
                record["json_output"] = str(resolve_output_path(json_output))
            else:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                record["returncode"] = completed.returncode
                record["stdout"] = completed.stdout
                record["stderr"] = completed.stderr
                record["json_output"] = str(resolve_output_path(json_output))
                if completed.returncode == 0 and resolve_output_path(json_output).exists():
                    payload = _load_json(json_output)
                    record["status"] = "ok"
                    record["metrics"] = payload.get("metrics", {})
                    record["correctness"] = payload.get("correctness", {})
                    record["device"] = payload.get("device")
                    if _is_better_result(record, best_result):
                        best_result = _result_record_from_bench_record(record, candidate)
                else:
                    record["status"] = "failed"
        records.append(record)
    iteration_payload = {
        "iteration": iteration_index,
        "results": records,
        "best_result": best_result,
    }
    write_json(iteration_dir / "results.json", iteration_payload)
    return iteration_payload


def _result_record_from_bench_record(record: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    kernel_ms = _metric_median(metrics, "kernel_ms")
    return {
        "status": record.get("status"),
        "candidate": candidate_to_dict(candidate),
        "config": dict(candidate.config),
        "config_tag": candidate.tag,
        "metrics": metrics,
        "correctness": record.get("correctness"),
        "device": record.get("device"),
        "kernel_ms_median": kernel_ms,
        "json_output": record.get("json_output"),
    }


def _metric_median(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if not isinstance(value, dict):
        return None
    median = value.get("median")
    return float(median) if median is not None else None


def _is_better_result(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> bool:
    if lhs is None:
        return False
    if rhs is None:
        return True
    lhs_ok = bool((lhs.get("correctness") or {}).get("correct", lhs.get("status") == "ok"))
    rhs_ok = bool((rhs.get("correctness") or {}).get("correct", rhs.get("status") == "ok"))
    if lhs_ok != rhs_ok:
        return lhs_ok
    lhs_kernel = lhs.get("kernel_ms_median")
    rhs_kernel = rhs.get("kernel_ms_median")
    if lhs_kernel is None:
        return False
    if rhs_kernel is None:
        return True
    return float(lhs_kernel) < float(rhs_kernel)


def _candidate_from_result(result: dict[str, Any]) -> Candidate:
    return Candidate(
        operation=result["candidate"]["operation"],
        config=dict(result["config"]),
        strategy=result["candidate"]["strategy"],
        reason=result["candidate"]["reason"],
        implementation=result["candidate"]["implementation"],
    )


def _maybe_profile_and_diagnose(
    spec: OperatorSpec,
    adapter: OperatorAdapter,
    best_result: dict[str, Any],
    options: AgentRunOptions,
    iteration_dir: Path,
    config_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if options.mode != "tune":
        return None
    candidate = _candidate_from_result(best_result)
    profile_dir = ensure_directory(iteration_dir / "profile")
    profile_command = adapter.profile_command(
        spec=spec,
        candidate=candidate,
        options=options,
        out_dir=profile_dir,
        tag=candidate.tag,
    )
    profile_record = {
        "command": profile_command,
        "command_text": quote_command(profile_command),
    }
    if options.dry_run:
        profile_record["status"] = "dry_run"
        write_json(profile_dir / "profile_record.json", profile_record)
        return {
            "operation": spec.operation,
            "spec_name": spec.name,
            "bottleneck": "mixed / inconclusive",
            "metrics": {},
            "next_experiments": adapter.build_iteration_candidates(
                spec,
                candidate,
                {"bottleneck": "mixed / inconclusive"},
                config_payload,
                limit=3,
            ),
            "profile_record": profile_record,
        }
    completed = subprocess.run(profile_command, cwd=PROJECT_DIR, text=True, capture_output=True, check=False)
    profile_record["returncode"] = completed.returncode
    profile_record["stdout"] = completed.stdout
    profile_record["stderr"] = completed.stderr
    write_json(profile_dir / "profile_record.json", profile_record)
    if completed.returncode != 0:
        return None
    ncu_details = profile_dir / f"{candidate.tag}_ncu_details.txt"
    if not ncu_details.exists():
        matches = list(profile_dir.glob("*_ncu_details.txt"))
        if matches:
            ncu_details = matches[0]
    if not ncu_details.exists():
        return None
    diagnosis = analyze_ncu_report(spec, ncu_details, best_result.get("config"))
    diagnosis["profile_record"] = profile_record
    return diagnosis


def _write_report(run_dir: Path, spec: OperatorSpec, summary: dict[str, Any]) -> None:
    best = summary.get("best_result")
    diagnosis = summary.get("diagnosis")
    lines = [
        f"# Triton 算子 Agent 报告：{spec.name}",
        "",
        f"- 算子：`{spec.operation}`",
        f"- dtype：`{spec.dtype}`",
        f"- 问题规模：`{json.dumps(spec.problem, ensure_ascii=False)}`",
        f"- 运行模式：`{summary['run']['mode']}`",
        f"- dry-run：`{'yes' if summary['run']['dry_run'] else 'no'}`",
        "",
        "## 迭代概览",
        "",
    ]
    for iteration in summary.get("iterations", []):
        lines.append(
            f"- iter_{iteration['iteration']:02d}：评估 `{len(iteration.get('results', []))}` 个候选"
        )
    if best is not None:
        lines.extend(
            [
                "",
                "## 当前最优候选",
                "",
                f"- config：`{best.get('config_tag')}`",
                f"- kernel_ms.median：`{_fmt_metric(best.get('kernel_ms_median'))}`",
                f"- 结果文件：`{best.get('json_output', '-')}`",
            ]
        )
    if diagnosis is not None:
        lines.extend(
            [
                "",
                "## 诊断结论",
                "",
                f"- bottleneck：`{diagnosis.get('bottleneck', 'mixed / inconclusive')}`",
            ]
        )
        for item in diagnosis.get("next_experiments", [])[:3]:
            lines.append(
                f"- 下一步：`{item.get('id', 'exp')}`，{item.get('reason', '')}，建议修改 `{json.dumps(item.get('changes', {}), ensure_ascii=False)}`"
            )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_memory(memory_path: Path, spec: OperatorSpec, summary: dict[str, Any]) -> None:
    resolved = resolve_output_path(memory_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": []}
    if resolved.exists():
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    entries = payload.setdefault("entries", [])
    best = summary.get("best_result")
    entries.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "spec_name": spec.name,
            "operation": spec.operation,
            "problem": dict(spec.problem),
            "hardware": dict(spec.hardware),
            "mode": summary["run"]["mode"],
            "dry_run": summary["run"]["dry_run"],
            "best_config_tag": best.get("config_tag") if best else None,
            "best_kernel_ms_median": best.get("kernel_ms_median") if best else None,
            "diagnosis": summary.get("diagnosis", {}).get("bottleneck")
            if isinstance(summary.get("diagnosis"), dict)
            else None,
            "run_dir": str(resolve_output_path(summary["run"]["run_dir"])),
        }
    )
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _run_metadata(options: AgentRunOptions) -> dict[str, Any]:
    return {
        "mode": options.mode,
        "dry_run": options.dry_run,
        "emit_nvtx": options.emit_nvtx,
        "device_index": options.device_index,
        "run_dir": str(resolve_output_path(options.run_dir)),
        "memory_path": str(resolve_output_path(options.memory_path)),
    }


def _effective_warmup(spec: OperatorSpec, options: AgentRunOptions) -> int:
    return options.warmup if options.warmup is not None else spec.budgets.warmup


def _effective_repeat(spec: OperatorSpec, options: AgentRunOptions) -> int:
    return options.repeat if options.repeat is not None else spec.budgets.repeat


def _default_matmul_like_configs(space: dict[str, list[int]]) -> list[dict[str, int]]:
    defaults = {
        "BLOCK_M": space.get("BLOCK_M", [64, 128]),
        "BLOCK_N": space.get("BLOCK_N", [64, 128]),
        "BLOCK_K": space.get("BLOCK_K", [32, 64]),
        "GROUP_M": space.get("GROUP_M", [4, 8]),
        "num_warps": space.get("num_warps", [2, 4, 8]),
        "num_stages": space.get("num_stages", [2, 3, 4]),
    }
    configs: list[dict[str, int]] = []
    for block_m in defaults["BLOCK_M"]:
        for block_n in defaults["BLOCK_N"]:
            for block_k in defaults["BLOCK_K"]:
                for group_m in defaults["GROUP_M"]:
                    for num_warps in defaults["num_warps"]:
                        for num_stages in defaults["num_stages"]:
                            configs.append(
                                {
                                    "BLOCK_M": int(block_m),
                                    "BLOCK_N": int(block_n),
                                    "BLOCK_K": int(block_k),
                                    "GROUP_M": int(group_m),
                                    "num_warps": int(num_warps),
                                    "num_stages": int(num_stages),
                                }
                            )
    return configs


def _matmul_like_priority(config: dict[str, int], problem: dict[str, int]) -> tuple[float, int]:
    target_m = max(64, min(128, int(problem.get("m", 128))))
    target_n = max(64, min(128, int(problem.get("n", 128))))
    block_area_delta = abs(config["BLOCK_M"] - target_m) + abs(config["BLOCK_N"] - target_n)
    pipeline_penalty = abs(config["num_stages"] - 3) + abs(config["num_warps"] - 4)
    return (float(block_area_delta + pipeline_penalty), config["GROUP_M"])


def _step_config(base: dict[str, int], allowed: dict[str, list[int]], key: str, delta: int) -> dict[str, int]:
    updated = dict(base)
    values = allowed.get(key)
    current = updated.get(key)
    if values is None or current is None:
        return updated
    sorted_values = sorted(set(int(item) for item in values))
    if current not in sorted_values:
        sorted_values.append(int(current))
        sorted_values.sort()
    index = sorted_values.index(int(current))
    index = max(0, min(len(sorted_values) - 1, index + delta))
    updated[key] = int(sorted_values[index])
    return updated


def _extract_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def _maybe_increment(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value) + 1


def _maybe_decrement(value: int | None) -> int | None:
    if value is None or value <= 1:
        return None
    return int(value) - 1


def _double_if_possible(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value) * 2


def _halve_if_possible(value: int | None) -> int | None:
    if value is None or value <= 1:
        return None
    return max(1, int(value) // 2)


def _toggle_group_m(value: int | None) -> int | None:
    if value is None:
        return None
    return 8 if int(value) <= 4 else 4


def _filtered_changes(base: dict[str, int], changes: dict[str, int | None]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in changes.items():
        if value is None:
            continue
        if base.get(key) == value:
            continue
        result[key] = int(value)
    return result


def _layernorm_block_candidates(hidden: int) -> list[int]:
    base = _next_power_of_2(max(128, hidden))
    values = {min(1024, base), min(1024, max(128, base // 2))}
    return sorted(values)


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f} ms"

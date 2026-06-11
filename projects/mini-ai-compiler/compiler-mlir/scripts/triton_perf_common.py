#!/usr/bin/env python3
"""Shared helpers for Triton performance workflows in compiler-mlir."""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PROJECT_DIR = Path(__file__).resolve().parents[1]
PERF_DIR = PROJECT_DIR / "perf"
CONFIG_KEYS = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "num_warps", "num_stages")


@dataclass(frozen=True)
class LinearReluCase:
    name: str
    description: str
    operation: str
    dtype: str
    m: int
    n: int
    k: int
    data_profile: str
    seed: int
    abs_tol: float
    rel_tol: float
    layout: dict[str, Any]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_DIR / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_linear_relu_case(path: str | Path) -> LinearReluCase:
    payload = load_json(path)
    problem = payload.get("problem", {})
    tolerance = payload.get("tolerance", {})
    return LinearReluCase(
        name=str(payload.get("name", "unnamed_case")),
        description=str(payload.get("description", "")),
        operation=str(payload.get("operation", "fused_linear_relu")),
        dtype=str(payload.get("dtype", "float32")),
        m=int(problem.get("m", 0)),
        n=int(problem.get("n", 0)),
        k=int(problem.get("k", 0)),
        data_profile=str(payload.get("data_profile", "gaussian")),
        seed=int(payload.get("seed", 0)),
        abs_tol=float(tolerance.get("abs", 1.0e-4)),
        rel_tol=float(tolerance.get("rel", 1.0e-4)),
        layout=dict(payload.get("layout", {})),
    )


def load_config_payload(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def normalize_config_values(config: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"missing Triton config keys: {', '.join(missing)}")
    for key in CONFIG_KEYS:
        values[key] = int(config[key])
    return values


def select_config(payload: dict[str, Any], source: str, overrides: dict[str, int]) -> dict[str, int]:
    base = payload.get(source)
    if not isinstance(base, dict):
        raise ValueError(f"config source '{source}' missing in payload")
    merged = dict(base)
    merged.update(overrides)
    return normalize_config_values(merged)


def iter_sweep_configs(payload: dict[str, Any]) -> list[dict[str, int]]:
    default = select_config(payload, "default", {})
    sweep = payload.get("sweep", {})
    if not isinstance(sweep, dict):
        raise ValueError("config payload field 'sweep' must be an object")
    keys: list[str] = []
    values: list[list[int]] = []
    for key in CONFIG_KEYS:
        raw = sweep.get(key)
        if raw is None:
            continue
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"sweep key '{key}' must be a non-empty array")
        keys.append(key)
        values.append([int(item) for item in raw])
    if not keys:
        return [default]
    configs: list[dict[str, int]] = []
    for combination in itertools.product(*values):
        candidate = dict(default)
        for key, value in zip(keys, combination):
            candidate[key] = value
        configs.append(normalize_config_values(candidate))
    return configs


def summarize_timings(timings_ms: list[float], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "min": min(timings_ms) if timings_ms else None,
        "mean": statistics.fmean(timings_ms) if timings_ms else None,
        "median": statistics.median(timings_ms) if timings_ms else None,
        "max": max(timings_ms) if timings_ms else None,
        "timings_ms": timings_ms,
    }


def summarize_scalar(value: float | None, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": unit}


def effective_gflops(case: LinearReluCase, kernel_ms: float | None) -> float | None:
    if kernel_ms is None or kernel_ms <= 0.0:
        return None
    flop_count = 2.0 * case.m * case.n * case.k + 2.0 * case.m * case.n
    return flop_count / (kernel_ms * 1.0e-3) / 1.0e9


def effective_gbps(case: LinearReluCase, kernel_ms: float | None, element_bytes: int) -> float | None:
    if kernel_ms is None or kernel_ms <= 0.0:
        return None
    bytes_moved = (
        case.m * case.k + case.n * case.k + case.n + case.m * case.n
    ) * float(element_bytes)
    return bytes_moved / (kernel_ms * 1.0e-3) / 1.0e9


def max_abs_err(lhs: Any, rhs: Any) -> float:
    return float((lhs - rhs).abs().max().item())


def max_rel_err(lhs: Any, rhs: Any) -> float:
    denom = rhs.abs().clamp_min(1.0e-8)
    return float(((lhs - rhs).abs() / denom).max().item())


def config_tag(config: dict[str, int]) -> str:
    return (
        f"bm{config['BLOCK_M']}_bn{config['BLOCK_N']}_bk{config['BLOCK_K']}"
        f"_gm{config['GROUP_M']}_w{config['num_warps']}_s{config['num_stages']}"
    )


def quote_command(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(text: str) -> str:
    if not text:
        return "''"
    special = set(" \t\n\"'`$&|;<>()[]{}*?!")
    if any(ch in special for ch in text):
        return "'" + text.replace("'", "'\"'\"'") + "'"
    return text


def ensure_directory(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def detect_device_name(torch_module: Any, device_index: int) -> str:
    props = torch_module.cuda.get_device_properties(device_index)
    return f"{props.name} (sm_{props.major}{props.minor})"


def validate_case(case: LinearReluCase) -> None:
    if case.operation != "fused_linear_relu":
        raise ValueError(
            f"unsupported operation '{case.operation}'; only fused_linear_relu is implemented"
        )
    if case.dtype not in {"float32", "f32"}:
        raise ValueError(f"unsupported dtype '{case.dtype}'; only float32 is implemented")
    for name, value in (("m", case.m), ("n", case.n), ("k", case.k)):
        if value <= 0:
            raise ValueError(f"problem dimension {name} must be > 0")


@contextlib.contextmanager
def nvtx_range(torch_module: Any, name: str, enabled: bool) -> Iterator[None]:
    if not enabled or not hasattr(torch_module.cuda, "nvtx"):
        yield
        return
    torch_module.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch_module.cuda.nvtx.range_pop()


def require_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable '{name}' is not set")
    return value


def median_metric(metrics: dict[str, Any], name: str) -> float | None:
    metric = metrics.get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("median")
    return float(value) if value is not None else None


def compare_close(abs_err: float, rel_err: float, abs_tol: float, rel_tol: float) -> bool:
    return abs_err <= abs_tol or rel_err <= rel_tol or math.isclose(
        abs_err, 0.0, abs_tol=abs_tol
    )

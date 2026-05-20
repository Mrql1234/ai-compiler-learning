#!/usr/bin/env python3
"""Run compiler-mlir performance cases across comparable GPU backends."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class BackendResult:
    name: str
    kind: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    result: float | None
    timings_ms: list[float]
    metrics: dict[str, Any]
    json_path: Path
    artifacts: dict[str, Any]
    correct: bool | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def latency(self, key: str) -> float | None:
        if not self.timings_ms:
            return None
        if key == "min":
            return min(self.timings_ms)
        if key == "max":
            return max(self.timings_ms)
        if key == "mean":
            return statistics.fmean(self.timings_ms)
        if key == "median":
            return statistics.median(self.timings_ms)
        raise ValueError(f"unknown latency key: {key}")

    def metric(self, metric_name: str, key: str) -> float | None:
        metric = self.metrics.get(metric_name)
        if not isinstance(metric, dict):
            return None
        value = metric.get(key)
        return float(value) if value is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a perf case against compiler-generated, hand CUDA, or "
            "third-party library backends and emit comparable JSON results."
        )
    )
    parser.add_argument("case", type=Path, help="Perf case JSON file")
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Backend name to run; defaults to enabled backends in the case",
    )
    parser.add_argument(
        "--backend-command",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="Command used by an external backend; can be repeated",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for JSON and artifact outputs",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=PROJECT_DIR / "build",
        help="compiler-mlir build directory",
    )
    parser.add_argument(
        "--gpu-runner",
        type=Path,
        default=None,
        help="Override mini-compiler-gpu-runner path",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--metric",
        default="kernel_ms",
        help="Metric shown in the final table; use latency_ms for legacy v0 runs",
    )
    parser.add_argument(
        "--ptxas-cmd-options",
        default="",
        help="Forwarded to the MLIR NVVM lowering pipeline",
    )
    return parser.parse_args()


def load_case(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_backend_commands(values: list[str]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Expected --backend-command NAME=COMMAND, got: {value}")
        name, command = value.split("=", 1)
        commands[name] = shlex.split(command)
    return commands


def substitute_command_templates(
    command: list[str], replacements: dict[str, str]
) -> tuple[list[str], bool]:
    """Replace simple {name} placeholders in a backend command."""
    formatted: list[str] = []
    used_template = False
    for part in command:
        next_part = part
        for name, value in replacements.items():
            token = "{" + name + "}"
            if token in next_part:
                used_template = True
                next_part = next_part.replace(token, value)
        formatted.append(next_part)
    return formatted, used_template


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_DIR / candidate


def selected_backends(case: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    backends = case.get("backends", [])
    if names:
        requested = set(names)
        selected = [backend for backend in backends if backend.get("name") in requested]
        missing = requested - {backend.get("name") for backend in selected}
        if missing:
            raise SystemExit(f"Unknown backend(s): {', '.join(sorted(missing))}")
        return selected
    return [backend for backend in backends if backend.get("enabled", False)]


def extract_float(stdout: str) -> float | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1])
    except ValueError:
        return None


def summarize_timings(timings_ms: list[float], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "min": min(timings_ms) if timings_ms else None,
        "mean": statistics.fmean(timings_ms) if timings_ms else None,
        "median": statistics.median(timings_ms) if timings_ms else None,
        "max": max(timings_ms) if timings_ms else None,
        "timings_ms": timings_ms,
    }


def load_runner_json(
    path: Path,
) -> tuple[float | None, list[float], dict[str, Any], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = payload.get("result")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    timings = [float(value) for value in payload.get("timings_ms", [])]
    if not timings:
        invoke_metric = metrics.get("invoke_ms")
        if isinstance(invoke_metric, dict):
            timings = [float(value) for value in invoke_metric.get("timings_ms", [])]
        legacy_latency = payload.get("latency_ms")
        if not timings and isinstance(legacy_latency, dict):
            timings = [
                float(value) for value in legacy_latency.get("timings_ms", [])
            ]
    return (float(result) if result is not None else None, timings, metrics, payload)


def metric_from_payload(
    metrics: dict[str, Any],
    timings_ms: list[float],
    metric_name: str,
    key: str,
) -> float | None:
    if metric_name == "latency_ms":
        if not timings_ms:
            return None
        if key == "min":
            return min(timings_ms)
        if key == "max":
            return max(timings_ms)
        if key == "mean":
            return statistics.fmean(timings_ms)
        if key == "median":
            return statistics.median(timings_ms)
        raise ValueError(f"unknown latency key: {key}")
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get(key)
    return float(value) if value is not None else None


def write_backend_json(result: BackendResult) -> None:
    payload = {
        "backend": result.name,
        "kind": result.kind,
        "command": result.command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "result": result.result,
        "correct": result.correct,
        "latency_ms": {
            "min": result.latency("min"),
            "mean": result.latency("mean"),
            "median": result.latency("median"),
            "max": result.latency("max"),
        },
        "timings_ms": result.timings_ms,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }
    with result.json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_mlir_backend(
    case: dict[str, Any],
    backend: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> BackendResult:
    runner = args.gpu_runner or args.build_dir / "bin" / "mini-compiler-gpu-runner"
    backend_name = backend["name"]
    json_path = run_dir / f"{backend_name}.json"
    lowered_path = run_dir / f"{backend_name}_lowered.mlir"
    command = [
        str(runner),
        str(resolve_project_path(backend.get("input", case["input"]))),
        "-e",
        backend.get("entry_function", case.get("entry_function", "run")),
        f"--entry-point-result={backend.get('result_type', case.get('result_type', 'f32'))}",
        f"--kernel-backend={backend.get('kernel_backend', backend_name)}",
        f"--gpu-chip={backend.get('gpu_chip', 'sm_86')}",
        f"--cubin-format={backend.get('cubin_format', 'fatbin')}",
        f"--opt-level={backend.get('opt_level', 3)}",
        f"--warmup={args.warmup}",
        f"--repeat={args.repeat}",
        f"--json-output={json_path}",
        f"--dump-lowered={lowered_path}",
    ]
    problem = case.get("problem", {})
    if problem:
        command.extend(
            [
                f"--problem-operation={problem.get('operation', 'linear_relu')}",
                f"--data-profile={problem.get('data_profile', 'deterministic')}",
                f"--m={problem.get('m', 2)}",
                f"--n={problem.get('n', 8)}",
                f"--k={problem.get('k', 4)}",
            ]
        )
    if backend.get("quantized", case.get("quantized", False)):
        command.append("--quantized")
    ptxas_options = backend.get("ptxas_cmd_options", args.ptxas_cmd_options)
    if ptxas_options:
        command.append(f"--ptxas-cmd-options={ptxas_options}")

    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        cwd=PROJECT_DIR,
    )
    result_value: float | None = extract_float(completed.stdout)
    timings_ms: list[float] = []
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {"lowered_mlir": str(lowered_path)}
    if completed.returncode == 0 and json_path.exists():
        result_value, timings_ms, metrics, runner_payload = load_runner_json(json_path)
        artifacts.update(runner_payload.get("artifacts", {}))
    backend_result = BackendResult(
        name=backend_name,
        kind=backend["kind"],
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        result=result_value,
        timings_ms=timings_ms,
        metrics=metrics,
        json_path=json_path,
        artifacts=artifacts,
    )
    write_backend_json(backend_result)
    return backend_result


def run_external_backend(
    case_path: Path,
    backend: dict[str, Any],
    commands: dict[str, list[str]],
    args: argparse.Namespace,
    run_dir: Path,
) -> BackendResult:
    backend_name = backend["name"]
    command = commands.get(backend_name)
    if command is None:
        raw_command = backend.get("command")
        if isinstance(raw_command, str):
            command = shlex.split(raw_command)
        elif isinstance(raw_command, list):
            command = [str(part) for part in raw_command]
    if not command:
        raise SystemExit(
            f"External backend '{backend_name}' needs --backend-command {backend_name}='<command>'"
        )

    json_path = run_dir / f"{backend_name}.json"
    replacements = {
        "backend": backend_name,
        "case": str(case_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "json_output": str(json_path.resolve()),
        "warmup": str(args.warmup),
        "repeat": str(args.repeat),
    }
    command, uses_template = substitute_command_templates(command, replacements)

    timings_ms: list[float] = []
    metrics: dict[str, Any] = {}
    last_completed: subprocess.CompletedProcess[str] | None = None
    result_value: float | None = None
    artifacts: dict[str, Any] = {}
    if uses_template or backend.get("run_once", False):
        start = time.perf_counter()
        last_completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            cwd=PROJECT_DIR,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        if last_completed.returncode == 0 and json_path.exists():
            result_value, timings_ms, metrics, runner_payload = load_runner_json(json_path)
            artifacts.update(runner_payload.get("artifacts", {}))
        elif last_completed.returncode == 0:
            result_value = extract_float(last_completed.stdout)
            timings_ms.append(duration_ms)
            metrics["invoke_ms"] = summarize_timings(timings_ms, "host_process_wall")
    else:
        total_runs = args.warmup + args.repeat
        for iteration in range(total_runs):
            start = time.perf_counter()
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                cwd=PROJECT_DIR,
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            last_completed = completed
            if completed.returncode != 0:
                break
            if iteration >= args.warmup:
                timings_ms.append(duration_ms)
        assert last_completed is not None
        result_value = extract_float(last_completed.stdout)
        metrics["invoke_ms"] = summarize_timings(timings_ms, "host_process_wall")

    assert last_completed is not None
    backend_result = BackendResult(
        name=backend_name,
        kind=backend["kind"],
        command=command,
        returncode=last_completed.returncode,
        stdout=last_completed.stdout,
        stderr=last_completed.stderr,
        result=result_value,
        timings_ms=timings_ms,
        metrics=metrics,
        json_path=json_path,
        artifacts=artifacts,
    )
    write_backend_json(backend_result)
    return backend_result


def apply_correctness(case: dict[str, Any], results: list[BackendResult]) -> None:
    tolerance = case.get("tolerance", {})
    abs_tol = float(tolerance.get("abs", 1.0e-5))
    rel_tol = float(tolerance.get("rel", 1.0e-5))
    reference = next(
        (result for result in results if result.succeeded and result.result is not None),
        None,
    )
    if reference is None:
        return
    for result in results:
        if not result.succeeded or result.result is None:
            result.correct = False
        else:
            result.correct = math.isclose(
                reference.result,
                result.result,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )
        write_backend_json(result)


def write_summary(case: dict[str, Any], run_dir: Path, results: list[BackendResult]) -> Path:
    summary_path = run_dir / "summary.json"
    payload = {
        "case": {
            "name": case.get("name"),
            "description": case.get("description"),
            "input": case.get("input"),
            "entry_function": case.get("entry_function", "run"),
            "result_type": case.get("result_type", "f32"),
            "problem": case.get("problem", {}),
            "tolerance": case.get("tolerance", {}),
        },
        "results": [
            {
                "backend": result.name,
                "kind": result.kind,
                "returncode": result.returncode,
                "result": result.result,
                "correct": result.correct,
                "latency_ms": {
                    "min": result.latency("min"),
                    "mean": result.latency("mean"),
                    "median": result.latency("median"),
                    "max": result.latency("max"),
                },
                "metrics": result.metrics,
                "json_path": str(result.json_path),
                "artifacts": result.artifacts,
            }
            for result in results
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return summary_path


def print_table(results: list[BackendResult], metric_name: str) -> None:
    print(f"metric: {metric_name}")
    print("backend        status   correct   result        median ms       mean ms")
    print("-------------  -------  --------  ------------  --------------  --------")
    for result in results:
        status = "ok" if result.succeeded else f"fail:{result.returncode}"
        correct = "-" if result.correct is None else ("yes" if result.correct else "no")
        value = "-" if result.result is None else f"{result.result:.9g}"
        median_ms = metric_from_payload(
            result.metrics, result.timings_ms, metric_name, "median"
        )
        mean_ms = metric_from_payload(
            result.metrics, result.timings_ms, metric_name, "mean"
        )
        median_text = "metric_missing" if median_ms is None else f"{median_ms:.3f}"
        mean_text = "-" if mean_ms is None else f"{mean_ms:.3f}"
        print(
            f"{result.name:<13}  {status:<7}  {correct:<8}  "
            f"{value:<12}  {median_text:>14}  {mean_text:>8}"
        )


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0")

    case = load_case(args.case)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or PROJECT_DIR / "perf" / "runs" / f"{case['name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    backend_commands = parse_backend_commands(args.backend_command)

    results: list[BackendResult] = []
    for backend in selected_backends(case, args.backend):
        if backend["kind"] == "mlir_gpu_runner":
            results.append(run_mlir_backend(case, backend, args, run_dir))
        elif backend["kind"] == "external":
            results.append(
                run_external_backend(args.case, backend, backend_commands, args, run_dir)
            )
        else:
            raise SystemExit(f"Unsupported backend kind: {backend['kind']}")

    apply_correctness(case, results)
    summary_path = write_summary(case, run_dir, results)
    print_table(results, args.metric)
    print(f"\nsummary: {summary_path}")
    return 0 if all(result.succeeded and result.correct is not False for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

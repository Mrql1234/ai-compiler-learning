#!/usr/bin/env python3
"""Benchmark and correctness harness for compiler-mlir CPU/GPU runners."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RunnerSummary:
    name: str
    command: list[str]
    result: float | None
    timings_ms: list[float]
    stdout: str
    stderr: str
    returncode: int

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    @property
    def warm_timings_ms(self) -> list[float]:
        return self.timings_ms

    def average_ms(self) -> float | None:
        if not self.timings_ms:
            return None
        return statistics.fmean(self.timings_ms)

    def median_ms(self) -> float | None:
        if not self.timings_ms:
            return None
        return statistics.median(self.timings_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run compiler-mlir CPU/GPU runners repeatedly, compare numerical "
            "results, and report simple latency statistics."
        )
    )
    parser.add_argument("input", type=Path, help="MLIR module to run")
    parser.add_argument(
        "--cpu-runner",
        type=Path,
        default=Path("./build/bin/mini-compiler-runner"),
        help="Path to the CPU runner binary",
    )
    parser.add_argument(
        "--gpu-runner",
        type=Path,
        default=Path("./build/bin/mini-compiler-gpu-runner"),
        help="Path to the GPU runner binary",
    )
    parser.add_argument(
        "--entry-function",
        default="run",
        help="Entry function name passed to both runners",
    )
    parser.add_argument(
        "--result-type",
        default="f32",
        choices=("f32", "void"),
        help="Entry result type expected by both runners",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations excluded from reported timings",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Measured iterations per runner",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for CPU/GPU result comparison",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-5,
        help="Relative tolerance for CPU/GPU result comparison",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Benchmark only the CPU path",
    )
    parser.add_argument(
        "--gpu-extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the GPU runner; can be repeated",
    )
    parser.add_argument(
        "--cpu-extra-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the CPU runner; can be repeated",
    )
    return parser.parse_args()


def extract_float(stdout: str) -> float | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1])
    except ValueError:
        return None


def build_command(
    runner: Path,
    input_file: Path,
    entry_function: str,
    result_type: str,
    extra_args: Iterable[str],
) -> list[str]:
    return [
        str(runner),
        str(input_file),
        "-e",
        entry_function,
        f"--entry-point-result={result_type}",
        *extra_args,
    ]


def run_runner(name: str, command: list[str], warmup: int, repeat: int) -> RunnerSummary:
    measured_ms: list[float] = []
    last_stdout = ""
    last_stderr = ""
    last_returncode = 0
    total_runs = warmup + repeat

    for iteration in range(total_runs):
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        last_stdout = completed.stdout
        last_stderr = completed.stderr
        last_returncode = completed.returncode
        if completed.returncode != 0:
            return RunnerSummary(
                name=name,
                command=command,
                result=extract_float(completed.stdout),
                timings_ms=measured_ms,
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
        if iteration >= warmup:
            measured_ms.append(duration_ms)

    return RunnerSummary(
        name=name,
        command=command,
        result=extract_float(last_stdout),
        timings_ms=measured_ms,
        stdout=last_stdout,
        stderr=last_stderr,
        returncode=last_returncode,
    )


def validate_paths(args: argparse.Namespace) -> None:
    for path in [args.input, args.cpu_runner]:
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")
    if not args.skip_gpu and not args.gpu_runner.exists():
        raise SystemExit(f"Missing GPU runner: {args.gpu_runner}")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0")


def report(summary: RunnerSummary) -> None:
    print(f"[{summary.name}]")
    print(f"  command : {' '.join(summary.command)}")
    print(f"  status  : {'ok' if summary.succeeded else f'failed ({summary.returncode})'}")
    if summary.result is not None:
        print(f"  result  : {summary.result:.9g}")
    if summary.timings_ms:
        print(f"  avg ms  : {summary.average_ms():.3f}")
        print(f"  med ms  : {summary.median_ms():.3f}")
        print(f"  min ms  : {min(summary.timings_ms):.3f}")
        print(f"  max ms  : {max(summary.timings_ms):.3f}")
    if summary.stderr.strip():
        print("  stderr  :")
        for line in summary.stderr.strip().splitlines():
            print(f"    {line}")


def compare_results(cpu: RunnerSummary, gpu: RunnerSummary, abs_tol: float, rel_tol: float) -> bool:
    if cpu.result is None or gpu.result is None:
        print("Correctness: skipped (missing numeric result)")
        return False
    ok = math.isclose(cpu.result, gpu.result, abs_tol=abs_tol, rel_tol=rel_tol)
    delta = abs(cpu.result - gpu.result)
    print(
        "Correctness: "
        f"{'PASS' if ok else 'FAIL'} "
        f"(cpu={cpu.result:.9g}, gpu={gpu.result:.9g}, |delta|={delta:.3g})"
    )
    return ok


def compare_speed(cpu: RunnerSummary, gpu: RunnerSummary) -> None:
    cpu_avg = cpu.average_ms()
    gpu_avg = gpu.average_ms()
    if cpu_avg is None or gpu_avg is None:
        print("Performance: skipped (missing timings)")
        return
    speedup = cpu_avg / gpu_avg if gpu_avg > 0.0 else float("inf")
    print(
        "Performance: "
        f"CPU avg {cpu_avg:.3f} ms vs GPU avg {gpu_avg:.3f} ms "
        f"(speedup {speedup:.3f}x)"
    )


def main() -> int:
    args = parse_args()
    validate_paths(args)

    cpu_summary = run_runner(
        "cpu",
        build_command(
            args.cpu_runner,
            args.input,
            args.entry_function,
            args.result_type,
            args.cpu_extra_arg,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    report(cpu_summary)
    if not cpu_summary.succeeded:
        return 1

    if args.skip_gpu:
        return 0

    gpu_summary = run_runner(
        "gpu",
        build_command(
            args.gpu_runner,
            args.input,
            args.entry_function,
            args.result_type,
            args.gpu_extra_arg,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    report(gpu_summary)
    if not gpu_summary.succeeded:
        return 1

    ok = compare_results(cpu_summary, gpu_summary, args.abs_tol, args.rel_tol)
    compare_speed(cpu_summary, gpu_summary)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

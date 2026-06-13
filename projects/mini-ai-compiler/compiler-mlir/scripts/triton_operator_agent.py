#!/usr/bin/env python3
"""Structured-entry Triton operator development and optimization agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from triton_operator_agent_lib import (
    ADAPTERS,
    DEFAULT_AGENT_MEMORY,
    AgentRunOptions,
    PROJECT_DIR,
    SUPPORTED_OPERATIONS,
    analyze_ncu_report,
    load_operator_spec,
    resolve_output_path,
    run_agent,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triton-first operator development and iterative optimization agent prototype"
    )
    parser.add_argument("--spec", type=Path, required=True, help="Path to an operator spec JSON")
    parser.add_argument(
        "--mode",
        choices=["plan", "tune", "analyze"],
        default="plan",
        help="plan: 只生成候选计划；tune: 执行候选并尝试迭代；analyze: 只分析 NCU 文本。",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output directory for plan, summary, report, and memory artifacts",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=DEFAULT_AGENT_MEMORY,
        help="Path to the persistent memory JSON file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print or store commands without executing them")
    parser.add_argument("--emit-nvtx", action="store_true", help="Pass NVTX markers to Triton benchmark/profile")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument(
        "--ncu-details",
        type=Path,
        default=None,
        help="Existing NCU details text. In analyze mode this is required.",
    )
    return parser.parse_args()


def default_run_dir(spec_name: str) -> Path:
    return PROJECT_DIR / "perf" / "runs" / "agent_runs" / spec_name


def main() -> int:
    args = parse_args()
    spec = load_operator_spec(args.spec)
    if spec.operation not in SUPPORTED_OPERATIONS:
        raise SystemExit(
            f"unsupported operation '{spec.operation}'. expected one of {sorted(SUPPORTED_OPERATIONS)}"
        )
    run_dir = resolve_output_path(args.run_dir or default_run_dir(spec.name))
    options = AgentRunOptions(
        mode=args.mode,
        dry_run=args.dry_run,
        emit_nvtx=args.emit_nvtx,
        device_index=args.device_index,
        run_dir=run_dir,
        memory_path=args.memory_path,
        max_candidates=args.max_candidates,
        max_iterations=args.max_iterations,
        warmup=args.warmup,
        repeat=args.repeat,
        ncu_details=args.ncu_details,
    )

    if args.mode == "analyze":
        if args.ncu_details is None:
            raise SystemExit("--ncu-details is required when --mode analyze")
        diagnosis = analyze_ncu_report(spec, args.ncu_details)
        write_json(run_dir / "analysis.json", diagnosis)
        print(f"spec: {spec.name}")
        print(f"operation: {spec.operation}")
        print(f"bottleneck: {diagnosis['bottleneck']}")
        print(f"analysis: {run_dir / 'analysis.json'}")
        return 0

    summary = run_agent(spec, options)
    best = summary.get("best_result")
    print(f"spec: {spec.name}")
    print(f"operation: {spec.operation}")
    print(f"mode: {args.mode}")
    print(f"adapter_execution: {'yes' if ADAPTERS[spec.operation].supports_execution else 'no'}")
    if best is not None:
        kernel = best.get("kernel_ms_median")
        metric = "-" if kernel is None else f"{kernel:.6f} ms"
        print(f"best_config: {best.get('config_tag', '-')}")
        print(f"best_kernel_ms: {metric}")
    diagnosis = summary.get("diagnosis")
    if isinstance(diagnosis, dict):
        print(f"bottleneck: {diagnosis.get('bottleneck', 'mixed / inconclusive')}")
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Sweep Triton fused linear + relu kernel configs and rank results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from triton_perf_common import (
    config_tag,
    ensure_directory,
    iter_sweep_configs,
    load_config_payload,
    load_linear_relu_case,
    median_metric,
    quote_command,
    resolve_input_path,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Triton fused linear + relu configs")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emit-nvtx", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def bench_script_path() -> Path:
    return PROJECT_DIR / "scripts" / "triton_linear_relu_bench.py"


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_for_candidate(
    args: argparse.Namespace, config: dict[str, int], json_output: Path
) -> list[str]:
    command = [
        sys.executable,
        str(bench_script_path()),
        "--case",
        str(resolve_input_path(args.case)),
        "--config",
        str(resolve_input_path(args.config)),
        "--config-source",
        "default",
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--device-index",
        str(args.device_index),
        "--json-output",
        str(json_output),
        "--BLOCK_M",
        str(config["BLOCK_M"]),
        "--BLOCK_N",
        str(config["BLOCK_N"]),
        "--BLOCK_K",
        str(config["BLOCK_K"]),
        "--GROUP_M",
        str(config["GROUP_M"]),
        "--num-warps",
        str(config["num_warps"]),
        "--num-stages",
        str(config["num_stages"]),
    ]
    if args.emit_nvtx:
        command.append("--emit-nvtx")
    return command


def write_markdown(out_dir: Path, case_name: str, ranking: list[dict[str, Any]]) -> None:
    lines = [
        f"# Triton sweep summary: {case_name}",
        "",
        "| rank | config | kernel_ms median | invoke_ms median | correct | json |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(ranking, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    item["config_tag"],
                    _fmt(item["kernel_ms_median"]),
                    _fmt(item["invoke_ms_median"]),
                    "yes" if item["correct"] else "no",
                    item["json_path"],
                ]
            )
            + " |"
        )
    (out_dir / "sweep_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("--warmup must be >= 0 and --repeat must be > 0")
    case = load_linear_relu_case(args.case)
    config_payload = load_config_payload(args.config)
    out_dir = ensure_directory(args.out)
    candidates = iter_sweep_configs(config_payload)
    if args.max_configs is not None:
        candidates = candidates[: args.max_configs]

    ranking: list[dict[str, Any]] = []
    executed: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        tag = config_tag(candidate)
        json_output = out_dir / f"{index:03d}_{tag}.json"
        command = command_for_candidate(args, candidate, json_output)
        record: dict[str, Any] = {
            "config": candidate,
            "config_tag": tag,
            "command": command,
            "json_path": str(json_output),
        }
        if args.dry_run:
            print(quote_command(command))
            executed.append(record)
            continue
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            cwd=PROJECT_DIR,
        )
        record.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode == 0 and json_output.exists():
            result = load_result(json_output)
            kernel_median = median_metric(result.get("metrics", {}), "kernel_ms")
            invoke_median = median_metric(result.get("metrics", {}), "invoke_ms")
            entry = {
                "config": candidate,
                "config_tag": tag,
                "kernel_ms_median": kernel_median,
                "invoke_ms_median": invoke_median,
                "correct": bool(result.get("correctness", {}).get("correct", False)),
                "json_path": str(json_output),
            }
            ranking.append(entry)
        executed.append(record)

    ranking.sort(
        key=lambda item: (
            item["kernel_ms_median"] is None,
            item["kernel_ms_median"] if item["kernel_ms_median"] is not None else float("inf"),
        )
    )
    summary = {
        "case": case.name,
        "case_path": str(resolve_input_path(args.case)),
        "config_path": str(resolve_input_path(args.config)),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "device_index": args.device_index,
        "dry_run": args.dry_run,
        "candidates": executed,
        "ranking": ranking[: args.top_k],
    }
    write_json(out_dir / "sweep_summary.json", summary)
    write_markdown(out_dir, case.name, ranking[: args.top_k])
    if ranking:
        write_json(out_dir / "best_config.json", ranking[0])
        print(f"best: {ranking[0]['config_tag']} kernel_ms={_fmt(ranking[0]['kernel_ms_median'])}")
    else:
        print("no successful sweep results")
    print(f"summary: {out_dir / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

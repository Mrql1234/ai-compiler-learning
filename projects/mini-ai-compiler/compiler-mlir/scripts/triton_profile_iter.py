#!/usr/bin/env python3
"""Profile a selected Triton fused linear + relu config with nsys and ncu."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from triton_perf_common import (
    ensure_directory,
    load_config_payload,
    quote_command,
    resolve_project_path,
    select_config,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one Triton config with nsys/ncu")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-source", choices=["default", "profile_target"], default="profile_target")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--tag", default="profile")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emit-nvtx", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--BLOCK_M", type=int, default=None)
    parser.add_argument("--BLOCK_N", type=int, default=None)
    parser.add_argument("--BLOCK_K", type=int, default=None)
    parser.add_argument("--GROUP_M", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    return parser.parse_args()


def bench_command(args: argparse.Namespace) -> list[str]:
    overrides = {
        "BLOCK_M": args.BLOCK_M,
        "BLOCK_N": args.BLOCK_N,
        "BLOCK_K": args.BLOCK_K,
        "GROUP_M": args.GROUP_M,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
    }
    overrides = {key: int(value) for key, value in overrides.items() if value is not None}
    config_payload = load_config_payload(args.config)
    _ = select_config(config_payload, args.config_source, overrides)
    output_json = ensure_directory(args.out) / f"{args.tag}_bench.json"
    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "triton_linear_relu_bench.py"),
        "--case",
        str(resolve_project_path(args.case)),
        "--config",
        str(resolve_project_path(args.config)),
        "--config-source",
        args.config_source,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--device-index",
        str(args.device_index),
        "--json-output",
        str(output_json),
    ]
    if args.emit_nvtx:
        command.append("--emit-nvtx")
    for key, value in overrides.items():
        if key.startswith("num_"):
            cli_name = "--" + key.replace("_", "-")
        else:
            cli_name = f"--{key}"
        command.extend([cli_name, str(value)])
    return command


def run_command(command: list[str], dry_run: bool) -> int:
    print(quote_command(command))
    if dry_run:
        return 0
    completed = subprocess.run(command, check=False, cwd=PROJECT_DIR)
    return completed.returncode


def export_text_reports(out_dir: Path, tag: str, dry_run: bool) -> None:
    nsys_rep = out_dir / f"{tag}_nsys.nsys-rep"
    ncu_rep = out_dir / f"{tag}_ncu.ncu-rep"
    commands = [
        [
            "nsys",
            "stats",
            "--force-export=true",
            "--report",
            "nvtx_pushpop_sum",
            str(nsys_rep),
        ],
        [
            "ncu",
            "--import",
            str(ncu_rep),
            "--page",
            "details",
        ],
        [
            "ncu",
            "--import",
            str(ncu_rep),
            "--page",
            "session",
        ],
    ]
    targets = [
        out_dir / f"{tag}_nsys_nvtx_summary.txt",
        out_dir / f"{tag}_ncu_details.txt",
        out_dir / f"{tag}_ncu_session.txt",
    ]
    for command, target in zip(commands, targets):
        print(quote_command(command) + f" > {target}")
        if dry_run:
            continue
        with target.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_DIR,
            )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    out_dir = ensure_directory(args.out)
    bench = bench_command(args)
    nsys_prefix = out_dir / f"{args.tag}_nsys"
    ncu_prefix = out_dir / f"{args.tag}_ncu"
    nsys_cmd = [
        str(PROJECT_DIR / "scripts" / "perf_profile_nsys.sh"),
        str(nsys_prefix),
        *bench,
    ]
    ncu_cmd = [
        str(PROJECT_DIR / "scripts" / "perf_profile_ncu.sh"),
        str(ncu_prefix),
        *bench,
    ]
    payload = {
        "tag": args.tag,
        "case_path": str(resolve_project_path(args.case)),
        "config_path": str(resolve_project_path(args.config)),
        "config_source": args.config_source,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "device_index": args.device_index,
        "dry_run": args.dry_run,
        "commands": {
            "bench": bench,
            "nsys": nsys_cmd,
            "ncu": ncu_cmd,
        },
        "artifacts": {
            "nsys_rep": str(nsys_prefix) + ".nsys-rep",
            "ncu_rep": str(ncu_prefix) + ".ncu-rep",
            "nsys_nvtx_summary": str(out_dir / f"{args.tag}_nsys_nvtx_summary.txt"),
            "ncu_details": str(out_dir / f"{args.tag}_ncu_details.txt"),
            "ncu_session": str(out_dir / f"{args.tag}_ncu_session.txt"),
        },
    }
    write_json(out_dir / f"{args.tag}_profile_plan.json", payload)
    if run_command(nsys_cmd, args.dry_run) != 0:
        return 1
    if run_command(ncu_cmd, args.dry_run) != 0:
        return 1
    export_text_reports(out_dir, args.tag, args.dry_run)
    print(f"profile plan: {out_dir / f'{args.tag}_profile_plan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Profile a selected Triton fused linear + relu config with nsys and ncu."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from triton_perf_common import (
    ensure_directory,
    load_config_payload,
    quote_command,
    resolve_input_path,
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
    parser.add_argument(
        "--ncu-nvtx-include",
        default="triton_linear_relu/benchmark",
        help="NVTX range filter passed to ncu when emit-nvtx is enabled",
    )
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--skip-nsys", action="store_true")
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
        str(resolve_input_path(args.case)),
        "--config",
        str(resolve_input_path(args.config)),
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


def run_command(command: list[str], dry_run: bool, env: dict[str, str] | None = None) -> int:
    print(quote_command(command))
    if dry_run:
        return 0
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    completed = subprocess.run(command, check=False, cwd=PROJECT_DIR, env=merged_env)
    return completed.returncode


def export_text_reports(
    out_dir: Path, tag: str, dry_run: bool, export_nsys: bool, export_ncu: bool
) -> None:
    commands: list[list[str]] = []
    targets: list[Path] = []
    if export_nsys:
        nsys_rep = out_dir / f"{tag}_nsys.nsys-rep"
        commands.append(
            [
                "nsys",
                "stats",
                "--force-export=true",
                "--report",
                "nvtx_pushpop_sum",
                str(nsys_rep),
            ]
        )
        targets.append(out_dir / f"{tag}_nsys_nvtx_summary.txt")
    if export_ncu:
        ncu_rep = out_dir / f"{tag}_ncu.ncu-rep"
        commands.extend(
            [
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
        )
        targets.extend(
            [
                out_dir / f"{tag}_ncu_details.txt",
                out_dir / f"{tag}_ncu_session.txt",
            ]
        )
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
    if args.skip_ncu and args.skip_nsys:
        raise SystemExit("cannot use --skip-ncu and --skip-nsys together")
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
        "case_path": str(resolve_input_path(args.case)),
        "config_path": str(resolve_input_path(args.config)),
        "config_source": args.config_source,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "device_index": args.device_index,
        "dry_run": args.dry_run,
        "skip_nsys": args.skip_nsys,
        "skip_ncu": args.skip_ncu,
        "ncu_nvtx_include": args.ncu_nvtx_include if args.emit_nvtx and not args.skip_ncu else None,
        "commands": {
            "bench": bench,
            "nsys": None if args.skip_nsys else nsys_cmd,
            "ncu": None if args.skip_ncu else ncu_cmd,
        },
        "artifacts": {
            "nsys_rep": None if args.skip_nsys else str(nsys_prefix) + ".nsys-rep",
            "ncu_rep": None if args.skip_ncu else str(ncu_prefix) + ".ncu-rep",
            "nsys_nvtx_summary": None
            if args.skip_nsys
            else str(out_dir / f"{args.tag}_nsys_nvtx_summary.txt"),
            "ncu_details": None if args.skip_ncu else str(out_dir / f"{args.tag}_ncu_details.txt"),
            "ncu_session": None if args.skip_ncu else str(out_dir / f"{args.tag}_ncu_session.txt"),
        },
    }
    write_json(out_dir / f"{args.tag}_profile_plan.json", payload)
    if not args.skip_nsys and run_command(nsys_cmd, args.dry_run) != 0:
        return 1
    ncu_env = None
    if args.emit_nvtx and args.ncu_nvtx_include and not args.skip_ncu:
        ncu_env = {"NCU_NVTX_INCLUDE": args.ncu_nvtx_include}
    if not args.skip_ncu and run_command(ncu_cmd, args.dry_run, env=ncu_env) != 0:
        return 1
    export_text_reports(
        out_dir,
        args.tag,
        args.dry_run,
        export_nsys=not args.skip_nsys,
        export_ncu=not args.skip_ncu,
    )
    print(f"profile plan: {out_dir / f'{args.tag}_profile_plan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

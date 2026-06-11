#!/usr/bin/env python3
"""Archive Triton iteration artifacts with manifests, summaries, and source snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_DIR / "perf" / "archive" / "triton_iterations"


@dataclass(frozen=True)
class GitMetadata:
    commit: str
    status: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Triton iteration artifacts")
    parser.add_argument("--kind", choices=["baseline", "sweep", "profile"], required=True)
    parser.add_argument("--iteration-name", required=True)
    parser.add_argument("--bench-json", type=Path)
    parser.add_argument("--sweep-summary", type=Path)
    parser.add_argument("--best-config", type=Path)
    parser.add_argument("--profile-plan", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--reference-bench-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> GitMetadata:
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
    )
    status_output = subprocess.check_output(
        ["git", "status", "--short"], cwd=PROJECT_DIR, text=True
    ).splitlines()
    return GitMetadata(commit=commit, status=status_output)


def rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def copy_sources(paths: list[Path], snapshot_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if not source.exists():
            continue
        rel = Path(rel_to_project(source))
        target = snapshot_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(
            {
                "path": str(rel),
                "snapshot_path": str(target.relative_to(snapshot_dir.parent)),
                "sha256": sha256_file(source),
            }
        )
    return copied


def extract_section_metrics(text: str, header_pattern: str) -> dict[str, Any]:
    sections = []
    header_regex = re.compile(header_pattern, re.MULTILINE)
    matches = list(header_regex.finditer(text))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[start:end])
    if not sections:
        return {}

    def score(section: str) -> tuple[float, float]:
        regs = extract_float(section, r"Registers Per Thread\s+register/thread\s+([0-9.]+)") or 0.0
        duration = extract_float(section, r"Duration\s+us\s+([0-9.]+)") or 0.0
        return (regs, duration)

    best = max(sections, key=score)
    return {
        "header": best.splitlines()[0].strip(),
        "duration_us": extract_float(best, r"Duration\s+us\s+([0-9.]+)"),
        "memory_throughput_gbps": extract_float(
            best, r"Memory Throughput\s+Gbyte/s\s+([0-9.]+)"
        ),
        "compute_sm_throughput_pct": extract_float(
            best, r"Compute \(SM\) Throughput\s+%\s+([0-9.]+)"
        ),
        "l2_hit_rate_pct": extract_float(best, r"L2 Hit Rate\s+%\s+([0-9.]+)"),
        "no_eligible_pct": extract_float(best, r"No Eligible\s+%\s+([0-9.]+)"),
        "eligible_warps_per_scheduler": extract_float(
            best, r"Eligible Warps Per Scheduler\s+warp\s+([0-9.]+)"
        ),
        "registers_per_thread": extract_float(
            best, r"Registers Per Thread\s+register/thread\s+([0-9.]+)"
        ),
        "dynamic_shared_memory_kib": extract_float(
            best, r"Dynamic Shared Memory Per Block\s+Kbyte/block\s+([0-9.]+)"
        ),
        "achieved_occupancy_pct": extract_float(best, r"Achieved Occupancy\s+%\s+([0-9.]+)"),
        "theoretical_occupancy_pct": extract_float(
            best, r"Theoretical Occupancy\s+%\s+([0-9.]+)"
        ),
        "block_size": extract_float(best, r"Block Size\s+([0-9.]+)"),
        "grid_size": extract_float(best, r"Grid Size\s+([0-9.]+)"),
        "shared_bank_conflicts": extract_float(
            best, r"results in ([0-9]+) bank conflicts"
        ),
        "shared_bank_conflict_wavefront_pct": extract_float(
            best, r"which represent\s+([0-9.]+)% of the overall"
        ),
        "short_scoreboard_cycles": extract_float(
            best, r"spends ([0-9.]+) cycles being stalled waiting for a scoreboard dependency"
        ),
    }


def extract_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def parse_nsys_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    result: dict[str, Any] = {}
    for range_name in ("prepare", "warmup", "benchmark"):
        match = re.search(
            rf"([0-9.]+)\s+([0-9]+)\s+[0-9]+\s+[0-9.]+\s+[0-9.]+\s+[0-9]+\s+[0-9]+\s+[0-9.]+\s+:triton_linear_relu/{range_name}",
            text,
        )
        if not match:
            continue
        result[range_name] = {
            "time_pct": float(match.group(1)),
            "total_time_ns": int(match.group(2)),
        }
    return result


def first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def ensure_output_dir(path: Path | None, iteration_name: str) -> Path:
    if path is None:
        path = ARCHIVE_ROOT / iteration_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def baseline_summary(bench_json: dict[str, Any]) -> tuple[dict[str, Any], str]:
    metrics = bench_json["metrics"]
    summary = {
        "iteration_kind": "baseline",
        "config": bench_json["config"],
        "case": bench_json["case"]["name"],
        "correct": bench_json["correctness"]["correct"],
        "kernel_ms_median": metrics["kernel_ms"]["median"],
        "invoke_ms_median": metrics["invoke_ms"]["median"],
        "effective_gflops": metrics["effective_gflops"]["value"],
        "effective_gbps": metrics["effective_gbps"]["value"],
    }
    analysis = "\n".join(
        [
            "# Iteration Analysis",
            "",
            "这一轮用于建立 Triton baseline。",
            "",
            f"- case: `{summary['case']}`",
            f"- config: `{bench_json['run']['config_tag']}`",
            f"- correctness: `{'yes' if summary['correct'] else 'no'}`",
            f"- kernel_ms.median: `{summary['kernel_ms_median']:.6f} ms`",
            f"- invoke_ms.median: `{summary['invoke_ms_median']:.6f} ms`",
            "",
            "用途：作为后续 sweep 和 profile 的性能对照。",
            "",
        ]
    )
    return summary, analysis


def sweep_summary(
    sweep_payload: dict[str, Any], best_payload: dict[str, Any], baseline_kernel_ms: float | None
) -> tuple[dict[str, Any], str]:
    ranking = sweep_payload.get("ranking", [])
    failed = [item for item in sweep_payload.get("candidates", []) if item.get("returncode") not in (0, None)]
    best_kernel_ms = best_payload["kernel_ms_median"]
    speedup = baseline_kernel_ms / best_kernel_ms if baseline_kernel_ms and best_kernel_ms else None
    summary = {
        "iteration_kind": "sweep",
        "candidate_count": len(sweep_payload.get("candidates", [])),
        "failed_candidate_count": len(failed),
        "best_config": best_payload["config"],
        "best_config_tag": best_payload["config_tag"],
        "best_kernel_ms_median": best_kernel_ms,
        "best_invoke_ms_median": best_payload["invoke_ms_median"],
        "speedup_vs_baseline": speedup,
        "top_ranked": ranking[:5],
        "failed_config_tags": [item["config_tag"] for item in failed],
    }
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
    analysis = "\n".join(
        [
            "# Iteration Analysis",
            "",
            "这一轮围绕 tile / pipeline 参数做 sweep。",
            "",
            f"- candidate_count: `{summary['candidate_count']}`",
            f"- failed_candidate_count: `{summary['failed_candidate_count']}`",
            f"- best_config_tag: `{summary['best_config_tag']}`",
            f"- best_kernel_ms.median: `{summary['best_kernel_ms_median']:.6f} ms`",
            f"- speedup_vs_baseline: `{speedup_text}`",
            "",
            f"结论：`128x128x32` 大 tile 仍然是当前主优区域，本轮名义最优配置是 `{summary['best_config_tag']}`。",
            "如果前几名差距非常小，仍需要补单点复测，再决定是否更新默认配置。",
            "",
        ]
    )
    return summary, analysis


def profile_summary(
    profile_dir: Path, profile_plan: dict[str, Any], reference_bench_json: Path | None
) -> tuple[dict[str, Any], str]:
    nsys_summary = parse_nsys_summary(profile_dir / f"{profile_plan['tag']}_nsys_nvtx_summary.txt")
    primary_ncu_details_path = profile_dir / f"{profile_plan['tag']}_ncu_details.txt"
    fallback_ncu_details_path = profile_dir / f"{profile_plan['tag']}_ncu_full_details.txt"
    ncu_details_path = first_existing_path(primary_ncu_details_path, fallback_ncu_details_path)
    ncu_metrics = extract_section_metrics(
        ncu_details_path.read_text(encoding="utf-8") if ncu_details_path is not None else "",
        r"^\s{2}kernel \([^\n]+$",
    )
    if not ncu_metrics and fallback_ncu_details_path.exists():
        ncu_details_path = fallback_ncu_details_path
        ncu_metrics = extract_section_metrics(
            fallback_ncu_details_path.read_text(encoding="utf-8"),
            r"^\s{2}kernel \([^\n]+$",
        )
    bench_payload = read_json(profile_dir / f"{profile_plan['tag']}_bench.json")
    reference_payload = read_json(reference_bench_json) if reference_bench_json else None
    summary = {
        "iteration_kind": "profile",
        "config_source": profile_plan["config_source"],
        "profiled_bench_kernel_ms_median": bench_payload["metrics"]["kernel_ms"]["median"],
        "profiled_bench_invoke_ms_median": bench_payload["metrics"]["invoke_ms"]["median"],
        "reference_bench_json": rel_to_project(reference_bench_json) if reference_bench_json else None,
        "ncu_details_path": rel_to_project(ncu_details_path) if ncu_details_path else None,
        "reference_kernel_ms_median": None
        if reference_payload is None
        else reference_payload["metrics"]["kernel_ms"]["median"],
        "reference_invoke_ms_median": None
        if reference_payload is None
        else reference_payload["metrics"]["invoke_ms"]["median"],
        "nsys_nvtx_ranges": nsys_summary,
        "ncu_triton_kernel": ncu_metrics,
    }
    analysis_lines = [
        "# Iteration Analysis",
        "",
        "这一轮做 `nsys + ncu` profile，用于定位下一轮优化方向。",
        "",
        f"- profiled_bench_kernel_ms.median: `{summary['profiled_bench_kernel_ms_median']:.6f} ms`",
        f"- profiled_bench_invoke_ms.median: `{summary['profiled_bench_invoke_ms_median']:.6f} ms`",
    ]
    if summary["reference_kernel_ms_median"] is not None:
        analysis_lines.extend(
            [
                f"- reference_kernel_ms.median: `{summary['reference_kernel_ms_median']:.6f} ms`",
                f"- reference_invoke_ms.median: `{summary['reference_invoke_ms_median']:.6f} ms`",
                "",
                "说明：profiled benchmark 数值会被 profiler 显著放大，性能结论应以 reference benchmark 为准。",
            ]
        )
    if ncu_metrics:
        shared_bank_conflicts = ncu_metrics.get("shared_bank_conflicts")
        if shared_bank_conflicts:
            conclusion = "结论：当前最突出的问题是 shared-memory bank conflict 和低 eligible warps，而不是 DRAM 带宽本身。"
        else:
            conclusion = "结论：当前目标 kernel 已不再出现显式 shared-memory bank-conflict 提示，下一轮更应关注低 occupancy 下的发射效率和资源压力平衡。"
        analysis_lines.extend(
            [
                f"- duration_us: `{ncu_metrics.get('duration_us')}`",
                f"- memory_throughput_gbps: `{ncu_metrics.get('memory_throughput_gbps')}`",
                f"- compute_sm_throughput_pct: `{ncu_metrics.get('compute_sm_throughput_pct')}`",
                f"- registers_per_thread: `{ncu_metrics.get('registers_per_thread')}`",
                f"- dynamic_shared_memory_kib: `{ncu_metrics.get('dynamic_shared_memory_kib')}`",
                f"- achieved_occupancy_pct: `{ncu_metrics.get('achieved_occupancy_pct')}`",
                f"- no_eligible_pct: `{ncu_metrics.get('no_eligible_pct')}`",
                f"- eligible_warps_per_scheduler: `{ncu_metrics.get('eligible_warps_per_scheduler')}`",
                f"- l2_hit_rate_pct: `{ncu_metrics.get('l2_hit_rate_pct')}`",
                f"- shared_bank_conflicts: `{int(ncu_metrics.get('shared_bank_conflicts') or 0)}`",
                f"- shared_bank_conflict_wavefront_pct: `{ncu_metrics.get('shared_bank_conflict_wavefront_pct')}`",
                f"- short_scoreboard_cycles: `{ncu_metrics.get('short_scoreboard_cycles')}`",
                "",
                conclusion,
            ]
        )
    analysis_lines.append("")
    return summary, "\n".join(analysis_lines)


def build_manifest(
    kind: str,
    iteration_name: str,
    output_dir: Path,
    source_snapshot: list[dict[str, Any]],
    artifact_paths: list[Path],
    commands: list[list[str]] | None,
) -> dict[str, Any]:
    git = git_metadata()
    return {
        "iteration_name": iteration_name,
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(PROJECT_DIR),
        "git": {
            "commit": git.commit,
            "dirty": bool(git.status),
            "status": git.status,
        },
        "archive_dir": rel_to_project(output_dir),
        "artifacts": [rel_to_project(path) for path in artifact_paths if path.exists()],
        "commands": commands,
        "source_snapshot": source_snapshot,
    }


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir, args.iteration_name)
    snapshot_dir = output_dir / "source_snapshot"

    if args.kind == "baseline":
        if args.bench_json is None:
            raise SystemExit("--bench-json is required for baseline")
        bench_path = (PROJECT_DIR / args.bench_json).resolve() if not args.bench_json.is_absolute() else args.bench_json
        bench_payload = read_json(bench_path)
        case_rel = Path("perf/cases") / f"{bench_payload['case']['name']}.json"
        config_rel = Path("perf/configs/triton_linear_relu_a10.json")
        sources = [
            PROJECT_DIR / "scripts/triton_linear_relu_bench.py",
            PROJECT_DIR / "scripts/triton_perf_common.py",
            PROJECT_DIR / case_rel,
            PROJECT_DIR / config_rel,
        ]
        source_snapshot = copy_sources(sources, snapshot_dir)
        summary, analysis = baseline_summary(bench_payload)
        manifest = build_manifest(
            "baseline",
            args.iteration_name,
            output_dir,
            source_snapshot,
            [bench_path],
            commands=None,
        )
    elif args.kind == "sweep":
        if args.sweep_summary is None or args.best_config is None:
            raise SystemExit("--sweep-summary and --best-config are required for sweep")
        summary_path = (
            (PROJECT_DIR / args.sweep_summary).resolve()
            if not args.sweep_summary.is_absolute()
            else args.sweep_summary
        )
        best_path = (
            (PROJECT_DIR / args.best_config).resolve()
            if not args.best_config.is_absolute()
            else args.best_config
        )
        sweep_payload = read_json(summary_path)
        best_payload = read_json(best_path)
        baseline_path = PROJECT_DIR / "perf/runs/triton_iterations/iter_00_baseline.json"
        baseline_kernel_ms = None
        if baseline_path.exists():
            baseline_kernel_ms = read_json(baseline_path)["metrics"]["kernel_ms"]["median"]
        sources = [
            PROJECT_DIR / "scripts/triton_linear_relu_bench.py",
            PROJECT_DIR / "scripts/triton_perf_common.py",
            PROJECT_DIR / "scripts/triton_perf_sweep.py",
            PROJECT_DIR / "perf/configs/triton_linear_relu_a10.json",
            PROJECT_DIR / "perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json",
        ]
        source_snapshot = copy_sources(sources, snapshot_dir)
        summary, analysis = sweep_summary(sweep_payload, best_payload, baseline_kernel_ms)
        commands = []
        for candidate in sweep_payload.get("candidates", [])[:1]:
            if candidate.get("command"):
                commands.append(candidate["command"])
                break
        manifest = build_manifest(
            "sweep",
            args.iteration_name,
            output_dir,
            source_snapshot,
            [summary_path, best_path],
            commands or None,
        )
    else:
        if args.profile_plan is None or args.profile_dir is None:
            raise SystemExit("--profile-plan and --profile-dir are required for profile")
        profile_dir = (
            (PROJECT_DIR / args.profile_dir).resolve()
            if not args.profile_dir.is_absolute()
            else args.profile_dir
        )
        plan_path = (
            (PROJECT_DIR / args.profile_plan).resolve()
            if not args.profile_plan.is_absolute()
            else args.profile_plan
        )
        plan_payload = read_json(plan_path)
        reference_bench_path = None
        if args.reference_bench_json is not None:
            reference_bench_path = (
                (PROJECT_DIR / args.reference_bench_json).resolve()
                if not args.reference_bench_json.is_absolute()
                else args.reference_bench_json
            )
        sources = [
            PROJECT_DIR / "scripts/triton_linear_relu_bench.py",
            PROJECT_DIR / "scripts/triton_perf_common.py",
            PROJECT_DIR / "scripts/triton_profile_iter.py",
            PROJECT_DIR / "scripts/perf_profile_nsys.sh",
            PROJECT_DIR / "scripts/perf_profile_ncu.sh",
            PROJECT_DIR / "perf/configs/triton_linear_relu_a10.json",
            PROJECT_DIR / "perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json",
        ]
        source_snapshot = copy_sources(sources, snapshot_dir)
        summary, analysis = profile_summary(profile_dir, plan_payload, reference_bench_path)
        artifact_paths = [plan_path] + sorted(profile_dir.glob("*"))
        manifest = build_manifest(
            "profile",
            args.iteration_name,
            output_dir,
            source_snapshot,
            artifact_paths,
            [
                command
                for command in (
                    plan_payload.get("commands", {}).get("bench"),
                    plan_payload.get("commands", {}).get("nsys"),
                    plan_payload.get("commands", {}).get("ncu"),
                )
                if command
            ],
        )

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "metrics_summary.json", summary)
    write_text(output_dir / "analysis.md", analysis + "\n")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

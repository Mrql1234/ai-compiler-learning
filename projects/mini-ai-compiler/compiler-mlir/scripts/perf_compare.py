#!/usr/bin/env python3
"""Compare compiler-mlir perf run summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact perf comparison table.")
    parser.add_argument(
        "summary",
        type=Path,
        help="summary.json file or a run directory containing summary.json",
    )
    parser.add_argument(
        "--baseline",
        default="cutlass",
        help="Backend used as the gap baseline when present",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "summary.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latency(result: dict[str, Any], key: str = "median") -> float | None:
    value = result.get("latency_ms", {}).get(key)
    return float(value) if value is not None else None


def main() -> int:
    args = parse_args()
    summary = load_summary(args.summary)
    results = summary.get("results", [])
    if not results:
        raise SystemExit("No results found")

    baseline_result = next(
        (result for result in results if result.get("backend") == args.baseline),
        None,
    )
    baseline_latency = latency(baseline_result) if baseline_result else None
    if baseline_latency is None:
        successful = [result for result in results if latency(result) is not None]
        if not successful:
            raise SystemExit("No successful timed results found")
        baseline_result = min(successful, key=lambda result: latency(result) or float("inf"))
        baseline_latency = latency(baseline_result)

    case_name = summary.get("case", {}).get("name", "<unknown>")
    baseline_name = baseline_result.get("backend") if baseline_result else "<none>"
    print(f"case: {case_name}")
    print(f"baseline: {baseline_name}")
    print("backend        correct   median ms   mean ms    gap")
    print("-------------  --------  ----------  --------  ------")
    for result in results:
        median_ms = latency(result, "median")
        mean_ms = latency(result, "mean")
        correct = result.get("correct")
        correct_text = "-" if correct is None else ("yes" if correct else "no")
        median_text = "-" if median_ms is None else f"{median_ms:.3f}"
        mean_text = "-" if mean_ms is None else f"{mean_ms:.3f}"
        if median_ms is None or baseline_latency in (None, 0.0):
            gap_text = "-"
        else:
            gap_text = f"{median_ms / baseline_latency:.2f}x"
        print(
            f"{result.get('backend', '<unknown>'):<13}  {correct_text:<8}  "
            f"{median_text:>10}  {mean_text:>8}  {gap_text:>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

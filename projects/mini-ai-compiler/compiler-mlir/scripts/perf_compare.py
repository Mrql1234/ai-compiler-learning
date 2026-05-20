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
        default="cublas",
        help="Backend used as the gap baseline when present",
    )
    parser.add_argument(
        "--metric",
        default="kernel_ms",
        help=(
            "Metric used for comparison. Use kernel_ms for fair kernel timing, "
            "invoke_ms for host invocation timing, or latency_ms for legacy v0 runs."
        ),
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "summary.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_stats(result: dict[str, Any], metric: str) -> dict[str, Any] | None:
    metrics = result.get("metrics", {})
    if isinstance(metrics, dict):
        value = metrics.get(metric)
        if isinstance(value, dict):
            return value
    legacy = result.get(metric)
    if isinstance(legacy, dict):
        return legacy
    return None


def metric_value(result: dict[str, Any] | None, metric: str, key: str = "median") -> float | None:
    if result is None:
        return None
    stats = metric_stats(result, metric)
    if stats is None:
        return None
    value = stats.get(key)
    return float(value) if value is not None else None


def available_metric_names(results: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for result in results:
        metrics = result.get("metrics", {})
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, dict):
                    names.add(name)
        if isinstance(result.get("latency_ms"), dict):
            names.add("latency_ms")
    return sorted(names)


def main() -> int:
    args = parse_args()
    summary = load_summary(args.summary)
    results = summary.get("results", [])
    if not results:
        raise SystemExit("No results found")

    timed_results = [
        result for result in results if metric_value(result, args.metric) is not None
    ]
    if not timed_results:
        available_metrics = available_metric_names(results)
        suffix = (
            f" Available metrics: {', '.join(available_metrics)}"
            if available_metrics
            else ""
        )
        raise SystemExit(
            f"No timed results found for metric '{args.metric}'.{suffix}"
        )

    baseline_result = next(
        (result for result in results if result.get("backend") == args.baseline),
        None,
    )
    baseline_latency = metric_value(baseline_result, args.metric)
    if baseline_latency is None:
        baseline_result = min(
            timed_results,
            key=lambda result: metric_value(result, args.metric) or float("inf"),
        )
        baseline_latency = metric_value(baseline_result, args.metric)

    case_name = summary.get("case", {}).get("name", "<unknown>")
    baseline_name = baseline_result.get("backend") if baseline_result else "<none>"
    print(f"case: {case_name}")
    print(f"metric: {args.metric}")
    print(f"baseline: {baseline_name}")
    print("backend        correct   median ms   mean ms    gap")
    print("-------------  --------  ----------  --------  ------")
    for result in results:
        median_ms = metric_value(result, args.metric, "median")
        mean_ms = metric_value(result, args.metric, "mean")
        correct = result.get("correct")
        correct_text = "-" if correct is None else ("yes" if correct else "no")
        median_text = "metric_missing" if median_ms is None else f"{median_ms:.3f}"
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

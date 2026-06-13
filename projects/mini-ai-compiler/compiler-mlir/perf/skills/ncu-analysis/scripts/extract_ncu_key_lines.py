#!/usr/bin/env python3
"""Extract high-signal lines from Nsight Compute text exports."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_PATTERNS = [
    r"occupancy",
    r"\bsm\b",
    r"dram",
    r"\bl2\b",
    r"register",
    r"shared memory",
    r"warp stall",
    r"scoreboard",
    r"throughput",
    r"launch statistics",
    r"duration",
    r"eligible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract high-signal Nsight Compute lines from text exports."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Text files exported from ncu")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Extra case-insensitive regex pattern to include",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=1,
        help="Number of neighboring lines to include around each match",
    )
    return parser.parse_args()


def build_regex(patterns: list[str]) -> re.Pattern[str]:
    combined = "|".join(f"(?:{pattern})" for pattern in patterns)
    return re.compile(combined, re.IGNORECASE)


def collect_hits(lines: list[str], regex: re.Pattern[str], context: int) -> list[int]:
    selected: set[int] = set()
    for index, line in enumerate(lines):
        if regex.search(line):
            for neighbor in range(max(0, index - context), min(len(lines), index + context + 1)):
                selected.add(neighbor)
    return sorted(selected)


def print_file(path: Path, context: int, regex: re.Pattern[str]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").replace("\ufeff", "").splitlines()
    hits = collect_hits(lines, regex, context)
    print(f"===== {path} =====")
    if not hits:
        print("(no matches)")
        return
    previous = -2
    for index in hits:
        if index > previous + 1:
            print("--")
        print(f"{index + 1:>6}: {lines[index]}")
        previous = index


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    regex = build_regex(DEFAULT_PATTERNS + args.pattern)
    for path in args.paths:
        print_file(path, args.context, regex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

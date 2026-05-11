#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./scripts/perf_profile_nsys.sh <output-prefix> <command> [args...]

Example:
  ./scripts/perf_profile_nsys.sh perf/runs/nsys_gpu_demo \
    ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
      --warmup=10 --repeat=100 --cubin-format=fatbin
EOF
  exit 1
fi

OUTPUT_PREFIX="$1"
shift

if ! command -v nsys >/dev/null 2>&1; then
  echo "[perf-nsys] missing nsys executable" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  -o "${OUTPUT_PREFIX}" \
  "$@"

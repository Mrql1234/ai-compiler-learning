#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'EOF'
Usage:
  ./scripts/perf_profile_ncu.sh <output-prefix> <command> [args...]

Example:
  ./scripts/perf_profile_ncu.sh perf/runs/ncu_gpu_demo \
    ./build/bin/mini-compiler-gpu-runner test/gpu_runner_demo.mlir \
      --warmup=5 --repeat=20 --cubin-format=fatbin
EOF
  exit 1
fi

OUTPUT_PREFIX="$1"
shift

if ! command -v ncu >/dev/null 2>&1; then
  echo "[perf-ncu] missing ncu executable" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

EXTRA_ARGS=()
if [[ -n "${NCU_NVTX_INCLUDE:-}" ]]; then
  EXTRA_ARGS+=(--nvtx --nvtx-include "${NCU_NVTX_INCLUDE}")
fi

ncu \
  --target-processes all \
  --set full \
  "${EXTRA_ARGS[@]}" \
  -o "${OUTPUT_PREFIX}" \
  "$@"

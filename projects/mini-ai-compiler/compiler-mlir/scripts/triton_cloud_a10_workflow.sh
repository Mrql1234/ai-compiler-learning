#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/triton_cloud_a10_workflow.sh <stage>

Stages:
  preflight  Check cloud A10 dependencies and profiling permissions.
  smoke      Run the small Triton smoke benchmark case.
  baseline   Run the main Triton baseline benchmark case.
  sweep      Sweep Triton tile / pipeline configs.
  profile    Run nsys + ncu for the selected profile target config.
  all        Run preflight + smoke + baseline + sweep + profile.

Environment overrides:
  DRY_RUN=1
  DEVICE_INDEX=0
  CASE_SMOKE=perf/cases/triton_linear_relu_f32_m128_n128_k128.json
  CASE_MAIN=perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json
  CONFIG=perf/configs/triton_linear_relu_a10.json
  SMOKE_WARMUP=5
  SMOKE_REPEAT=20
  BASELINE_WARMUP=10
  BASELINE_REPEAT=50
  SWEEP_WARMUP=10
  SWEEP_REPEAT=50
  PROFILE_WARMUP=1
  PROFILE_REPEAT=2
  PROFILE_CONFIG_SOURCE=profile_target
  PROFILE_EMIT_NVTX=1
  PROFILE_SKIP_NCU=0
  PROFILE_SKIP_NSYS=0
  PROFILE_TAG=iter_02_pipeline
  RUN_ROOT=perf/runs/triton_iterations
  PROFILE_ROOT=perf/profiles/triton_iterations
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

STAGE="$1"
DRY_RUN="${DRY_RUN:-0}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"

CASE_SMOKE="${CASE_SMOKE:-perf/cases/triton_linear_relu_f32_m128_n128_k128.json}"
CASE_MAIN="${CASE_MAIN:-perf/cases/triton_linear_relu_f32_m1024_n1024_k1024.json}"
CONFIG="${CONFIG:-perf/configs/triton_linear_relu_a10.json}"

SMOKE_WARMUP="${SMOKE_WARMUP:-5}"
SMOKE_REPEAT="${SMOKE_REPEAT:-20}"
BASELINE_WARMUP="${BASELINE_WARMUP:-10}"
BASELINE_REPEAT="${BASELINE_REPEAT:-50}"
SWEEP_WARMUP="${SWEEP_WARMUP:-10}"
SWEEP_REPEAT="${SWEEP_REPEAT:-50}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_REPEAT="${PROFILE_REPEAT:-2}"
PROFILE_CONFIG_SOURCE="${PROFILE_CONFIG_SOURCE:-profile_target}"
PROFILE_EMIT_NVTX="${PROFILE_EMIT_NVTX:-1}"
PROFILE_SKIP_NCU="${PROFILE_SKIP_NCU:-0}"
PROFILE_SKIP_NSYS="${PROFILE_SKIP_NSYS:-0}"
PROFILE_TAG="${PROFILE_TAG:-iter_02_pipeline}"

RUN_ROOT="${RUN_ROOT:-perf/runs/triton_iterations}"
PROFILE_ROOT="${PROFILE_ROOT:-perf/profiles/triton_iterations}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-${RUN_ROOT}/smoke_m128.json}"
BASELINE_OUTPUT="${BASELINE_OUTPUT:-${RUN_ROOT}/iter_00_baseline.json}"
SWEEP_OUTPUT="${SWEEP_OUTPUT:-${RUN_ROOT}/iter_01_tile}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-${PROFILE_ROOT}/iter_02_pipeline}"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_command() {
  local tool="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ command -v ${tool}"
    return 0
  fi
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "[triton-a10] missing required command: ${tool}" >&2
    exit 1
  fi
  command -v "${tool}"
}

run_preflight() {
  require_command python3
  require_command nvidia-smi
  require_command nsys
  require_command ncu
  run_cmd nvidia-smi
  run_cmd python3 -c \
    "import json, torch, triton; print(json.dumps({'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'triton': triton.__version__}, ensure_ascii=False, indent=2))"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ grep RmProfilingAdminOnly /proc/driver/nvidia/params"
    return 0
  fi

  local profiling_line
  profiling_line="$(grep RmProfilingAdminOnly /proc/driver/nvidia/params || true)"
  if [[ -z "${profiling_line}" ]]; then
    echo "[triton-a10] failed to read /proc/driver/nvidia/params" >&2
    exit 1
  fi
  echo "${profiling_line}"
  if [[ "${profiling_line}" != *"RmProfilingAdminOnly: 0"* ]]; then
    if [[ "${PROFILE_SKIP_NCU}" == "1" ]]; then
      echo "[triton-a10] GPU performance counters are locked; continuing with PROFILE_SKIP_NCU=1." >&2
      return 0
    fi
    echo "[triton-a10] GPU performance counters are locked; ncu profiling cannot run." >&2
    echo "[triton-a10] rerun with PROFILE_SKIP_NCU=1 to keep benchmark + nsys steps." >&2
    exit 1
  fi
}

run_smoke() {
  run_cmd python3 ./scripts/triton_linear_relu_bench.py \
    --case "${CASE_SMOKE}" \
    --config "${CONFIG}" \
    --config-source default \
    --warmup "${SMOKE_WARMUP}" \
    --repeat "${SMOKE_REPEAT}" \
    --device-index "${DEVICE_INDEX}" \
    --json-output "${SMOKE_OUTPUT}"
}

run_baseline() {
  run_cmd python3 ./scripts/triton_linear_relu_bench.py \
    --case "${CASE_MAIN}" \
    --config "${CONFIG}" \
    --config-source default \
    --warmup "${BASELINE_WARMUP}" \
    --repeat "${BASELINE_REPEAT}" \
    --device-index "${DEVICE_INDEX}" \
    --json-output "${BASELINE_OUTPUT}"
}

run_sweep() {
  run_cmd python3 ./scripts/triton_perf_sweep.py \
    --case "${CASE_MAIN}" \
    --config "${CONFIG}" \
    --warmup "${SWEEP_WARMUP}" \
    --repeat "${SWEEP_REPEAT}" \
    --device-index "${DEVICE_INDEX}" \
    --out "${SWEEP_OUTPUT}"
}

run_profile() {
  local -a command=(
    python3 ./scripts/triton_profile_iter.py
    --case "${CASE_MAIN}"
    --config "${CONFIG}"
    --config-source "${PROFILE_CONFIG_SOURCE}"
    --warmup "${PROFILE_WARMUP}"
    --repeat "${PROFILE_REPEAT}"
    --device-index "${DEVICE_INDEX}"
    --tag "${PROFILE_TAG}"
    --out "${PROFILE_OUTPUT}"
  )
  if [[ "${PROFILE_EMIT_NVTX}" == "1" ]]; then
    command+=(--emit-nvtx)
  fi
  if [[ "${PROFILE_SKIP_NCU}" == "1" ]]; then
    command+=(--skip-ncu)
  fi
  if [[ "${PROFILE_SKIP_NSYS}" == "1" ]]; then
    command+=(--skip-nsys)
  fi
  run_cmd "${command[@]}"
}

case "${STAGE}" in
  preflight)
    run_preflight
    ;;
  smoke)
    run_smoke
    ;;
  baseline)
    run_baseline
    ;;
  sweep)
    run_sweep
    ;;
  profile)
    run_profile
    ;;
  all)
    run_preflight
    run_smoke
    run_baseline
    run_sweep
    run_profile
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[triton-a10] unknown stage: ${STAGE}" >&2
    usage >&2
    exit 1
    ;;
esac

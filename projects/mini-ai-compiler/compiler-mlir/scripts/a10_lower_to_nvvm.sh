#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MINI_BUILD="${MINI_BUILD:-${PROJECT_DIR}/build}"
LLVM_MLIR_BUILD="${LLVM_MLIR_BUILD:-}"

INPUT_PATH="${1:-${PROJECT_DIR}/test/gpu_runner_demo.mlir}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/a10_nvvm}"
GPU_CHIP="${GPU_CHIP:-}"
GPU_FEATURES="${GPU_FEATURES:-}"
OPT_LEVEL="${OPT_LEVEL:-3}"
CUBIN_FORMAT="${CUBIN_FORMAT:-isa}"
ENTRY_FUNCTION="${ENTRY_FUNCTION:-run}"
RESULT_TYPE="${RESULT_TYPE:-f32}"

MINI_OPT="${MINI_BUILD}/bin/mini-compiler-opt"
GPU_RUNNER="${MINI_BUILD}/bin/mini-compiler-gpu-runner"

resolve_llvm_mlir_build() {
  local cache_path="${MINI_BUILD}/CMakeCache.txt"
  local mlir_dir
  if [[ -f "${cache_path}" ]]; then
    mlir_dir="$(grep '^MLIR_DIR:' "${cache_path}" | cut -d= -f2- || true)"
    if [[ -n "${mlir_dir}" ]]; then
      dirname "$(dirname "$(dirname "${mlir_dir}")")"
      return 0
    fi
  fi
  return 1
}

resolve_gpu_chip() {
  if [[ -n "${GPU_CHIP}" ]]; then
    printf '%s\n' "${GPU_CHIP}"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.[:space:]')"
    if [[ -n "${compute_cap}" ]]; then
      printf 'sm_%s\n' "${compute_cap}"
      return 0
    fi
  fi

  printf 'sm_86\n'
}

if [[ -z "${LLVM_MLIR_BUILD}" ]]; then
  if ! LLVM_MLIR_BUILD="$(resolve_llvm_mlir_build)"; then
    cat >&2 <<EOF
[a10-nvvm] unable to determine LLVM_MLIR_BUILD automatically.

Set LLVM_MLIR_BUILD explicitly or configure ${MINI_BUILD} with matching LLVM_DIR/MLIR_DIR.
EOF
    exit 1
  fi
fi

GPU_CHIP="$(resolve_gpu_chip)"

LLVM_CACHE="${LLVM_MLIR_BUILD}/CMakeCache.txt"

if [[ ! -x "${MINI_OPT}" ]]; then
  echo "[a10-nvvm] missing mini-compiler-opt: ${MINI_OPT}" >&2
  exit 1
fi
if [[ ! -x "${GPU_RUNNER}" ]]; then
  echo "[a10-nvvm] missing mini-compiler-gpu-runner: ${GPU_RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${LLVM_CACHE}" ]]; then
  echo "[a10-nvvm] missing CMake cache: ${LLVM_CACHE}" >&2
  exit 1
fi
if ! grep -q '^LLVM_TARGETS_TO_BUILD:STRING=.*NVPTX' "${LLVM_CACHE}"; then
  echo "[a10-nvvm] LLVM build does not include NVPTX backend" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

GPU_IR_PATH="${OUTPUT_DIR}/00_gpu_launch.mlir"
ATTACHED_PATH="${OUTPUT_DIR}/10_nvvm_attached.mlir"
LOWERED_PATH="${OUTPUT_DIR}/20_nvvm_lowered.mlir"

ATTACH_OPTS="chip=${GPU_CHIP} O=${OPT_LEVEL}"
PIPELINE_OPTS="cubin-chip=${GPU_CHIP} cubin-format=${CUBIN_FORMAT} opt-level=${OPT_LEVEL}"

if [[ -n "${GPU_FEATURES}" ]]; then
  ATTACH_OPTS+=" features=${GPU_FEATURES}"
  PIPELINE_OPTS+=" cubin-features=${GPU_FEATURES}"
fi

echo "[a10-nvvm] input: ${INPUT_PATH}"
echo "[a10-nvvm] output dir: ${OUTPUT_DIR}"
echo "[a10-nvvm] gpu chip: ${GPU_CHIP}"
echo "[a10-nvvm] cubin format: ${CUBIN_FORMAT}"
echo "[a10-nvvm] opt level: ${OPT_LEVEL}"

"${MINI_OPT}" --mini-gpu-lowering "${INPUT_PATH}" > "${GPU_IR_PATH}"
echo "[a10-nvvm] wrote ${GPU_IR_PATH}"

"${MINI_OPT}" "${GPU_IR_PATH}" \
  --nvvm-attach-target="${ATTACH_OPTS}" \
  > "${ATTACHED_PATH}"
echo "[a10-nvvm] wrote ${ATTACHED_PATH}"

"${GPU_RUNNER}" "${INPUT_PATH}" \
  -e "${ENTRY_FUNCTION}" \
  --entry-point-result="${RESULT_TYPE}" \
  --gpu-chip="${GPU_CHIP}" \
  --cubin-format="${CUBIN_FORMAT}" \
  --opt-level="${OPT_LEVEL}" \
  --dump-lowered="${LOWERED_PATH}" \
  >/dev/null
echo "[a10-nvvm] wrote ${LOWERED_PATH}"

cat <<EOF
[a10-nvvm] done.

Artifacts:
  - ${GPU_IR_PATH}
  - ${ATTACHED_PATH}
  - ${LOWERED_PATH}

Notes:
  - 00/10 两阶段产物来自 mini-compiler-opt 的分阶段导出。
  - 20 阶段产物来自 mini-compiler-gpu-runner 的完整本地 NVVM lowering。
  - CUBIN_FORMAT=isa is best for inspection/debugging.
  - CUBIN_FORMAT=fatbin is the more realistic setting for cloud execution.
  - Default GPU chip is ${GPU_CHIP}; override with GPU_CHIP=... if needed.
EOF

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MINI_BUILD="${MINI_BUILD:-${PROJECT_DIR}/build}"
LLVM_MLIR_BUILD="${LLVM_MLIR_BUILD:-/home/ql/code/llvm_clang_static_analyzer/build-mlir}"

INPUT_PATH="${1:-${PROJECT_DIR}/test/gpu_prep.mlir}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/a10_nvvm}"
GPU_CHIP="${GPU_CHIP:-sm_86}"
GPU_FEATURES="${GPU_FEATURES:-}"
OPT_LEVEL="${OPT_LEVEL:-3}"
CUBIN_FORMAT="${CUBIN_FORMAT:-isa}"

MINI_OPT="${MINI_BUILD}/bin/mini-compiler-opt"
MLIR_OPT="${LLVM_MLIR_BUILD}/bin/mlir-opt"
LLVM_CACHE="${LLVM_MLIR_BUILD}/CMakeCache.txt"

if [[ ! -x "${MINI_OPT}" ]]; then
  echo "[a10-nvvm] missing mini-compiler-opt: ${MINI_OPT}" >&2
  exit 1
fi
if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "[a10-nvvm] missing mlir-opt: ${MLIR_OPT}" >&2
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

"${MLIR_OPT}" "${GPU_IR_PATH}" \
  --nvvm-attach-target="${ATTACH_OPTS}" \
  > "${ATTACHED_PATH}"
echo "[a10-nvvm] wrote ${ATTACHED_PATH}"

"${MLIR_OPT}" "${ATTACHED_PATH}" \
  --gpu-lower-to-nvvm-pipeline="${PIPELINE_OPTS}" \
  > "${LOWERED_PATH}"
echo "[a10-nvvm] wrote ${LOWERED_PATH}"

cat <<EOF
[a10-nvvm] done.

Artifacts:
  - ${GPU_IR_PATH}
  - ${ATTACHED_PATH}
  - ${LOWERED_PATH}

Notes:
  - CUBIN_FORMAT=isa is best for inspection/debugging.
  - CUBIN_FORMAT=fatbin is the more realistic setting for cloud execution.
  - Default A10 chip is set to ${GPU_CHIP}; override with GPU_CHIP=... if needed.
EOF

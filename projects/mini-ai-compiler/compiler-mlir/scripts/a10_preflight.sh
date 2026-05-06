#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MINI_BUILD="${MINI_BUILD:-${PROJECT_DIR}/build}"
LLVM_MLIR_BUILD="${LLVM_MLIR_BUILD:-}"

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

if [[ -z "${LLVM_MLIR_BUILD}" ]]; then
  if ! LLVM_MLIR_BUILD="$(resolve_llvm_mlir_build)"; then
    cat >&2 <<EOF
[preflight] unable to determine LLVM_MLIR_BUILD automatically.

Set LLVM_MLIR_BUILD explicitly or configure ${MINI_BUILD} with matching LLVM_DIR/MLIR_DIR.
EOF
    exit 1
  fi
fi

MINI_OPT="${MINI_BUILD}/bin/mini-compiler-opt"
MLIR_OPT="${LLVM_MLIR_BUILD}/bin/mlir-opt"
LLVM_CACHE="${LLVM_MLIR_BUILD}/CMakeCache.txt"

echo "[preflight] project dir: ${PROJECT_DIR}"
echo "[preflight] mini build: ${MINI_BUILD}"
echo "[preflight] llvm/mlir build: ${LLVM_MLIR_BUILD}"

for tool in "${MINI_OPT}" "${MLIR_OPT}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "[preflight] missing executable: ${tool}" >&2
    exit 1
  fi
done

if [[ ! -f "${LLVM_CACHE}" ]]; then
  echo "[preflight] missing CMake cache: ${LLVM_CACHE}" >&2
  exit 1
fi

if ! "${MLIR_OPT}" --help | grep -q "gpu-lower-to-nvvm-pipeline"; then
  echo "[preflight] mlir-opt does not expose gpu-lower-to-nvvm-pipeline" >&2
  exit 1
fi

if ! grep -q '^LLVM_TARGETS_TO_BUILD:STRING=.*NVPTX' "${LLVM_CACHE}"; then
  cat >&2 <<'EOF'
[preflight] NVPTX backend is not present in this LLVM build.

On the cloud A10 machine, rebuild LLVM with NVPTX enabled, e.g.:

  -DLLVM_TARGETS_TO_BUILD="host;NVPTX"

Then rebuild MLIR against that LLVM build.
EOF
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[preflight] GPU inventory:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "[preflight] nvidia-smi not found; this is acceptable on the local no-GPU machine."
fi

cat <<'EOF'
[preflight] A10 NVVM toolchain checks passed.

Recommended next step:

  ./scripts/a10_lower_to_nvvm.sh test/gpu_prep.mlir
EOF

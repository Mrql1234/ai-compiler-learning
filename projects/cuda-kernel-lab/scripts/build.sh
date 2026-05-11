#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
if [[ -z "${CUDA_ARCHITECTURES}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')"
fi
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-86}"

cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" -G Ninja \
  -DCUDA_KERNEL_LAB_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
cmake --build "${PROJECT_DIR}/build" -j

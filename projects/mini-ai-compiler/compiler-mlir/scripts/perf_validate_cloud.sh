#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build}"
LLVM_DIR="${LLVM_DIR:-/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/llvm}"
MLIR_DIR="${MLIR_DIR:-/home/ql/toolchains/llvm_clang_static_analyzer/build/lib/cmake/mlir}"
MINI_CUDA_ARCHITECTURES="${MINI_CUDA_ARCHITECTURES:-86}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
RUN_ID="${RUN_ID:-gpu_runner_demo_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-perf/runs/${RUN_ID}}"

cmake -S . -B "${BUILD_DIR}" \
  -G Ninja \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DMINI_CUDA_ARCHITECTURES="${MINI_CUDA_ARCHITECTURES}"

cmake --build "${BUILD_DIR}" -j2

python3 ./scripts/perf_run.py perf/cases/gpu_runner_demo.json \
  --backend mlir_nvvm \
  --backend cuda_hand \
  --backend cublas \
  --metric kernel_ms \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --run-dir "${RUN_DIR}"

python3 ./scripts/perf_compare.py \
  --metric kernel_ms \
  "${RUN_DIR}/summary.json" \
  | tee "${RUN_DIR}/compare_kernel_ms.txt"

echo "cloud perf validation artifacts: ${RUN_DIR}"

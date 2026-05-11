#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${SCRIPT_DIR}/build.sh"
cd "${PROJECT_DIR}"
nsys profile --stats=true --output vector_add_nsys ./build/cuda_vector_add

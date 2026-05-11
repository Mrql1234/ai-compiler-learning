#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" -G Ninja
cmake --build "${PROJECT_DIR}/build" -j


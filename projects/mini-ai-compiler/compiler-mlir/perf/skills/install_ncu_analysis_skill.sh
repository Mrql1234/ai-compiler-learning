#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/ncu-analysis"
TARGET_ROOT="${1:-${HOME}/.codex/skills}"
TARGET_DIR="${TARGET_ROOT}/ncu-analysis"

mkdir -p "${TARGET_ROOT}"
if [[ -e "${TARGET_DIR}" ]]; then
  echo "target already exists: ${TARGET_DIR}"
  echo "please back it up or remove it manually before re-running this installer."
  exit 1
fi
cp -R "${SOURCE_DIR}" "${TARGET_DIR}"

echo "installed: ${TARGET_DIR}"

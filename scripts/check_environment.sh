#!/usr/bin/env bash
set -euo pipefail

echo "== Accelerateworld environment check =="

for command in nvidia-smi nvcc cmake; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "ERROR: ${command} is not available on PATH." >&2
    exit 1
  fi
done

nvidia-smi
echo
nvcc --version
echo
cmake --version | head -n 1

echo
echo "Environment check: PASS"

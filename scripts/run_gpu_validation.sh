#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${ACCELERATEWORLD_GPU_INDEX:-0}"
BUILD_DIR="${ACCELERATEWORLD_BUILD_DIR:-build/gpu-native}"
RESULT_ROOT="${ACCELERATEWORLD_RESULT_ROOT:-results/gpu-baselines}"
mkdir -p results

ARGS=(
  --gpu-index "${GPU_INDEX}"
  --build-dir "${BUILD_DIR}"
  --result-root "${RESULT_ROOT}"
)

if [[ "${ACCELERATEWORLD_SKIP_SANITIZER:-0}" == "1" ]]; then
  ARGS+=(--skip-sanitizer)
fi

if [[ "${ACCELERATEWORLD_ALLOW_NON_RTX:-0}" == "1" ]]; then
  ARGS+=(--allow-non-rtx)
fi

python3 scripts/run_gpu_baseline.py "${ARGS[@]}" 2>&1 | tee results/gpu-validation.log

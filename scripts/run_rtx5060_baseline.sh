#!/usr/bin/env bash
set -euo pipefail

WITH_AI=0
if [[ "${1:-}" == "--with-ai" ]]; then
  WITH_AI=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RESULT_DIR="${ACCELERATEWORLD_RESULT_DIR:-results}"
BUILD_DIR="${ACCELERATEWORLD_BUILD_DIR:-build/gpu}"
mkdir -p "${RESULT_DIR}"

for command in nvidia-smi nvcc cmake python compute-sanitizer; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "Required command not found: ${command}" >&2
    exit 2
  }
done

GPU_ENV="${RESULT_DIR}/rtx5060-environment.txt"
{
  echo "== RTX 5060 baseline environment =="
  date -u +"UTC: %Y-%m-%dT%H:%M:%SZ"
  echo "git: $(git rev-parse HEAD)"
  echo
  nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader
  echo
  nvcc --version
  echo
  cmake --version
  echo
  python --version
} 2>&1 | tee "${GPU_ENV}"

if ! grep -qi "RTX 5060" "${GPU_ENV}"; then
  echo "This baseline must run on an NVIDIA GeForce RTX 5060." >&2
  exit 3
fi

export ACCELERATEWORLD_CUDA_ARCH=120
export ACCELERATEWORLD_BUILD_DIR="${BUILD_DIR}"
export ACCELERATEWORLD_RESULT_DIR="${RESULT_DIR}"

bash scripts/run_gpu_validation.sh

"${BUILD_DIR}/bin/aw_device_query" | tee "${RESULT_DIR}/rtx5060-device-query.txt"
grep -qi "RTX 5060" "${RESULT_DIR}/rtx5060-device-query.txt"
grep -q "Compute capability: 12.0" "${RESULT_DIR}/rtx5060-device-query.txt"

python scripts/run_benchmark_suite.py \
  --build-dir "${BUILD_DIR}" \
  --output "${RESULT_DIR}/rtx5060-native.json"

if [[ "${WITH_AI}" -eq 1 ]]; then
  bash scripts/run_ai_gpu_validation.sh
fi

VERIFY_ARGS=(--result-dir "${RESULT_DIR}")
if [[ "${WITH_AI}" -eq 1 ]]; then
  VERIFY_ARGS+=(--require-ai)
fi
python scripts/verify_baseline_evidence.py "${VERIFY_ARGS[@]}"

python - "${RESULT_DIR}" "${WITH_AI}" <<'PY'
from __future__ import annotations
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
with_ai = sys.argv[2] == "1"
native = json.loads((result_dir / "rtx5060-native.json").read_text(encoding="utf-8"))
summary = {
    "schema_version": 1,
    "status": "PASS",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "target": "NVIDIA GeForce RTX 5060",
    "cuda_architecture": "sm_120",
    "native_success": bool(native.get("success")),
    "ai_validation_included": with_ai,
    "evidence": [
        "rtx5060-environment.txt",
        "rtx5060-device-query.txt",
        "gpu-validation.log",
        "rtx5060-native.json",
    ] + (["ai-gpu-validation.log"] if with_ai else []),
}
(result_dir / "rtx5060-baseline-summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

echo "RTX 5060 baseline: PASS"

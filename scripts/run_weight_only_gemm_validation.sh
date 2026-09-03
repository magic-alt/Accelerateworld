#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${ACCELERATEWORLD_RESULT_DIR:-results}"
mkdir -p "${RESULT_DIR}"

echo "== INT8 / INT4 weight-only GEMM physical validation =="
date -u +"UTC: %Y-%m-%dT%H:%M:%SZ"
nvidia-smi
python --version
python - <<'PY'
import torch
import triton
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "CUDA GPU required"
PY

pushd experiments/32_weight_only_gemm >/dev/null
python setup.py build_ext --inplace
python reference_test.py
popd >/dev/null

python experiments/32_weight_only_gemm/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --formats int8_sym,int4_sym \
  --granularity group \
  --group-size 64 \
  --warmup 5 \
  --iterations 20 \
  --json "${RESULT_DIR}/weight-only-gemm-validation.json"

echo "weight-only GEMM physical validation: PASS"

#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${ACCELERATEWORLD_RESULT_DIR:-results}"
mkdir -p "${RESULT_DIR}"
LOG_FILE="${RESULT_DIR}/ai-gpu-validation.log"

{
  echo "== Accelerateworld AI/GPU validation =="
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

  echo
  echo "== PyTorch CUDA Extension =="
  pushd experiments/10_pytorch_extension >/dev/null
  python setup.py build_ext --inplace
  python benchmark.py --elements 16777216 --iterations 50
  python autograd_test.py --elements 4096
  python fake_tensor_test.py
  python compile_test.py --rows 16 --cols 257
  python compile_benchmark.py --elements 4194304 --warmup 10 --iterations 30
  popd >/dev/null

  echo
  echo "== Triton =="
  python experiments/11_triton/vector_add.py --elements 16777216

  echo
  echo "== LLM kernels =="
  python experiments/12_llm_kernels/rmsnorm.py --rows 4096 --cols 4096 --dtype fp16

  echo
  echo "AI/GPU validation: PASS"
} 2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${ACCELERATEWORLD_RESULT_DIR:-results}/nsys-framework-trace"
mkdir -p "${RESULT_ROOT}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: Nsight Systems CLI 'nsys' is required on the self-hosted GPU runner." >&2
  exit 2
fi

nsys --version

pushd experiments/10_pytorch_extension >/dev/null
python setup.py build_ext --inplace
popd >/dev/null

# Steady-state capture: Inductor compilation and Triton autotuning happen before
# cudaProfilerStart(), so the report focuses on framework dispatch, CUDA APIs,
# kernel launches, synchronization and vendor/Triton/custom kernels.
python experiments/24_nsys_trace/run_trace.py \
  --mode steady \
  --elements 1048576 \
  --m 16 --n 4096 --k 4096 \
  --warmup 3 \
  --iterations 8 \
  --output-dir "${RESULT_ROOT}/steady"

# Full first-use capture: deliberately includes the first torch.compile/Inductor
# call and Triton autotune call. Use a smaller GEMM to keep the trace artifact
# practical while retaining the compiler/runtime transitions.
python experiments/24_nsys_trace/run_trace.py \
  --mode full \
  --elements 262144 \
  --m 16 --n 1024 --k 1024 \
  --warmup 1 \
  --iterations 2 \
  --output-dir "${RESULT_ROOT}/full"

#!/usr/bin/env bash
set -euo pipefail

ARCH="${ACCELERATEWORLD_CUDA_ARCH:-120}"
BUILD_DIR="${ACCELERATEWORLD_BUILD_DIR:-build/gpu}"
RESULT_DIR="${ACCELERATEWORLD_RESULT_DIR:-results}"
mkdir -p "${RESULT_DIR}"

LOG_FILE="${RESULT_DIR}/gpu-validation.log"

{
  echo "== Accelerateworld GPU validation =="
  date -u +"UTC: %Y-%m-%dT%H:%M:%SZ"
  echo "CUDA architecture: ${ARCH}"
  echo

  nvidia-smi
  echo
  nvcc --version
  echo

  cmake -S . -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DACCELERATEWORLD_CUDA_ARCHITECTURES="${ARCH}"
  cmake --build "${BUILD_DIR}" --parallel

  ctest --test-dir "${BUILD_DIR}" --output-on-failure

  echo
  echo "== Compute Sanitizer: memcheck =="
  compute-sanitizer --tool memcheck --error-exitcode 99 \
    "${BUILD_DIR}/bin/aw_vector_add" --elements 1048576 --iterations 2
  compute-sanitizer --tool memcheck --error-exitcode 99 \
    "${BUILD_DIR}/bin/aw_matmul" --size 128 --iterations 2

  echo
  echo "== Compute Sanitizer: racecheck / initcheck / synccheck =="
  compute-sanitizer --tool racecheck --error-exitcode 99 \
    "${BUILD_DIR}/bin/aw_matmul" --size 128 --iterations 1
  compute-sanitizer --tool initcheck --error-exitcode 99 \
    "${BUILD_DIR}/bin/aw_vector_add" --elements 1048576 --iterations 1
  compute-sanitizer --tool synccheck --error-exitcode 99 \
    "${BUILD_DIR}/bin/aw_matmul" --size 128 --iterations 1

  echo
  echo "== Benchmarks =="
  "${BUILD_DIR}/bin/aw_vector_add" --elements 33554432 --iterations 50
  "${BUILD_DIR}/bin/aw_matmul" --size 1024 --iterations 10

  echo
  echo "GPU validation: PASS"
} 2>&1 | tee "${LOG_FILE}"

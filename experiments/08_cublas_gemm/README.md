# Experiment 08 — cuBLAS SGEMM Baseline

After writing an educational GEMM, the next question is what a mature vendor library achieves on the same class of operation. This lab establishes a simple cuBLAS FP32 baseline.

## Goal

The executable allocates square FP32 A/B/C matrices and calls `cublasSgemm`. Inputs are constants (`0.5` and `0.25`), so every output element has an analytically known value:

```text
C[i,j] = N * 0.5 * 0.25 = N * 0.125
```

That makes correctness independent of a second GEMM implementation.

## Row-major caution

cuBLAS historically exposes column-major BLAS semantics. This experiment uses square matrices with uniform values, so the analytical correctness check is insensitive to transpose interpretation. When adapting this code to arbitrary row-major data, reason explicitly about operand order, leading dimensions and transpose flags rather than copying the call mechanically.

## Build and run

```bash
cmake --preset release
cmake --build --preset release-build --target aw_cublas_gemm
./build/release/bin/aw_cublas_gemm --size 1024 --iterations 30
```

A warm-up SGEMM is synchronized before CUDA-event timing. Throughput uses the conventional dense GEMM count `2*N^3` FLOPs.

## Interpreting the result

This is a library baseline, not a claim that SGEMM is the best precision mode for modern AI. Later experiments move through WMMA, cuBLASLt, BF16/TF32, FP8/FP4, CUTLASS and Triton. Keep this number because it demonstrates both the value of specialized libraries and how performance evolves as Tensor Core-friendly datatypes/algorithms enter the comparison.

## What to profile

Use Nsight Systems to confirm the library-call boundary and Nsight Compute later to inspect the selected kernel. Do not infer Tensor Core usage solely from a kernel name; hardware-counter evidence belongs in the Stage 7 profiling work.

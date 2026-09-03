# Experiment 09 — Tensor Core WMMA GEMM

This lab is the first direct Tensor Core programming exercise. It uses CUDA WMMA fragments with FP16 inputs and FP32 accumulation to show how a warp cooperatively computes a 16×16 matrix tile.

## Kernel structure

One 32-thread warp owns one output tile. The kernel creates WMMA fragments for matrix A, matrix B and the FP32 accumulator, then walks K in 16-element steps:

```text
load A 16x16 fragment
load B 16x16 fragment
mma_sync(acc, A, B, acc)
repeat over K
store 16x16 FP32 accumulator tile
```

The matrix size must be a positive multiple of 16. Device code is guarded for architectures with WMMA support.

## Precision contract

A and B are `half`; C is FP32. The multiply inputs therefore have FP16 representational limits while the running dot-product accumulator has FP32 storage. This mixed-precision structure is a precursor to the BF16/TF32/FP8 experiments later in Stage 3.

## Build and run

```bash
cmake --preset release
cmake --build --preset release-build --target aw_tensor_core_wmma
./build/release/bin/aw_tensor_core_wmma --size 1024 --iterations 20
```

Inputs are filled with 0.5 and 0.25, yielding an analytical expected result `N*0.125`. The benchmark reports CUDA-event latency and `2*N^3` GFLOP/s.

## What this implementation omits

Production Tensor Core GEMM needs more than calling `mma_sync`: shared-memory staging, vectorized/global-memory layouts, multiple warps per CTA, pipeline overlap, architecture-specific tile choices and epilogues all matter. This simple implementation is intentionally a conceptual bridge. Compare it with experiments 17–23 to see why cuBLASLt, CUTLASS and autotuned Triton operate at a higher abstraction/performance level.

## Profiling questions

Verify whether the expected Tensor Core instructions execute on your target, then inspect memory stalls and occupancy. A WMMA kernel can still be memory- or scheduling-limited even when the arithmetic instruction itself maps to Tensor Cores.

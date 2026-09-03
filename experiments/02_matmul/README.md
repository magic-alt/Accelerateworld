# Experiment 02 — Naive vs Tiled Matrix Multiply

This lab introduces data reuse and shared-memory tiling with a square FP32 GEMM. It is intentionally simple enough to audit and intentionally far from cuBLAS performance, making the effect of memory reuse visible.

## Goal

Both kernels compute `C=A×B` for `N×N` row-major matrices. `MatMulNaiveKernel` assigns one output element to one thread and rereads A/B operands from global memory inside the K loop. `MatMulTiledKernel` cooperatively stages 16×16 tiles in shared memory.

```text
naive: each output thread repeatedly loads global A/B

tiled: block loads A_tile + B_tile
       -> __syncthreads
       -> 16 multiply-add steps using shared data
       -> next K tile
```

## Tiling mechanics

The tiled kernel declares `tile_a[16][16]` and `tile_b[16][16]`. Boundary tiles load zero for out-of-range elements, so arbitrary positive N is supported even when N is not a multiple of 16. Two barriers per K tile ensure the tile is fully loaded before use and no thread overwrites it before every consumer finishes.

## FLOP model and timing

A dense N×N GEMM performs approximately `2*N^3` floating-point operations. The program times naive and tiled kernels independently with CUDA events and reports GFLOP/s plus the naive/tiled speedup.

```bash
cmake --preset release
cmake --build --preset release-build --target aw_matmul
./build/release/bin/aw_matmul --size 512 --iterations 10
```

For `N<=256`, a CPU O(N³) implementation is used as the correctness oracle. For larger N, the two independent GPU implementations are cross-checked to avoid making the host reference dominate the lab runtime.

## What this experiment proves

Shared-memory tiling can reduce redundant global loads because values staged by the block are reused across many multiply-adds. It does **not** prove the tile is optimal, that occupancy is ideal, or that the kernel approaches a vendor GEMM. Later WMMA, cuBLAS, cuBLASLt, CUTLASS and Triton experiments progressively add the machinery needed for production GEMM performance.

## Profiling questions

Compare global load efficiency, shared-memory throughput, arithmetic intensity and achieved occupancy. Try N values that do and do not align to 16. Then compare this lab with experiment 08: the performance gap between an educational tiled GEMM and cuBLAS is itself an important systems lesson.

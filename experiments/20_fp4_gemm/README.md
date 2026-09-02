# Blackwell FP4 / NVFP4-style cuBLASLt GEMM

This experiment studies the next Stage 3 precision boundary: 4-bit E2M1 Tensor Core GEMM with 16-element UE4M3 block scaling on Blackwell.

It compares three independently autotuned TN-layout paths under one deterministic workload:

1. BF16 input / FP32 accumulation and output;
2. tensorwide-scaled FP8 E4M3 / FP32 accumulation and output;
3. block-scaled FP4 E2M1 / FP32 accumulation and output.

The goal is not to label every E2M1 matmul as the full Transformer Engine NVFP4 training recipe. This lab implements the cuBLASLt block-scaled input path and also models NVFP4's two-level input scaling by combining a per-tensor FP32 global scale with per-16-element UE4M3 block scales. Transformer Engine adds training-specific policy such as stochastic rounding, 2D weight scaling, recipe state and delayed/current scaling choices that remain outside this experiment.

## Hardware gate

GPU Baseline v2 records the FP4 target only when both `fp4` and `blackwell` are present:

```text
RTX 20 / Turing   -> FP4 skipped
RTX 30 / Ampere   -> FP4 skipped
RTX 40 / Ada      -> FP4 skipped
RTX 50 / Blackwell-> FP4 run
```

The executable repeats a runtime compute-capability gate before any FP4 cuBLASLt call.

## cuBLASLt FP4 contract

CUDA 13 documents the Blackwell block-scaled FP4 path as:

- A/B data type: `CUDA_R_4F_E2M1`;
- scale mode: `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`;
- one UE4M3 scale for each 16 adjacent E2M1 values in the K dimension;
- compute type: `CUBLAS_COMPUTE_32F`;
- scale type: `CUDA_R_32F`;
- A transposed, B non-transposed (TN layout);
- 16-byte aligned matrix and scale pointers.

The scale-factor storage is not a simple row-major array. cuBLASLt uses a tiled scale layout. One 128x4 scale tile describes a 128x64 FP4 data region. The experiment explicitly packs logical `(outer, k_block)` scale factors into this layout and pads incomplete tiles with zeroes.

## Two-level NVFP4-style input scaling

For each input tensor:

```text
global_scale = global_amax / (448 * 6)
```

where 448 is E4M3 max finite and 6 is E2M1 max finite.

For every consecutive 16-value block:

```text
ideal_block_scale = block_amax / (6 * global_scale)
stored_block_scale = E4M3(ideal_block_scale)
fp4_value = E2M1(value / (global_scale * stored_block_scale))
```

cuBLASLt applies the UE4M3 block scales. The matmul `alpha` additionally carries `global_scale_A * global_scale_B`, reconstructing the two-level input scale in the final FP32 result.

The experiment records:

- global amax and FP32 global scale;
- minimum/maximum stored UE4M3 block scale;
- number of scale values near E4M3 max;
- E2M1 clipped / near-limit values;
- maximum quantize-dequantize reconstruction error;
- packed FP4 payload bytes;
- tiled scale-storage bytes.

## Correctness vs accuracy

The benchmark intentionally separates implementation correctness from expected low-precision error.

`Correctness reference` uses CPU FP64 dot products over the values reconstructed from the exact E2M1 payload plus stored UE4M3/global scales. This catches nibble packing, scale-tile layout, descriptor or dequantization mistakes.

`Accuracy reference` uses CPU FP64 dot products over the original FP32 master inputs. This measures the actual degradation caused by BF16, E4M3 or FP4 quantization.

Only the correctness error participates in PASS/FAIL. Accuracy degradation is evidence, not a bug by itself.

## Autotune path

Every mode independently performs:

```text
cuBLASLt descriptors
        -> workspace budget
        -> cublasLtMatmulAlgoGetHeuristic
        -> candidate warmup
        -> CUDA Event timing
        -> measured lowest-latency winner
```

The FP4 winner is not inherited from BF16 or FP8.

## Example

```bash
./build/release/bin/aw_fp4_gemm \
  --m 1024 --n 1024 --k 1024 \
  --heuristics 16 --warmup 2 --iterations 10 \
  --workspace-mb 32 --mode all
```

Modes are `all`, `bf16`, `e4m3`, and `fp4`.

## GPU Baseline v2

Canonical FP4 record:

```text
fp4_e2m1
requires: cuda + fp4 + blackwell
primary metric: fp4_e2m1_throughput
```

The canonical invocation also runs BF16 and E4M3 references so the evidence contains FP4 speedup and accuracy deltas under the same shape and TN layout.

## Evidence boundary

Hosted CUDA CI proves header/API availability, compile/link portability, manifest gating and CTest discovery. It does not prove that a physical Blackwell GPU selected an FP4 Tensor Core kernel, nor does it provide real scale distributions, algorithm winners, GFLOP/s or numerical error. Those belong in GPU Baseline v2 evidence collected on RTX 50 hardware.

Primary references: CUDA 13 cuBLASLt narrow-precision and 1D block-scaling documentation, CUDA Math API FP4 conversion intrinsics, and NVIDIA Transformer Engine NVFP4 documentation.

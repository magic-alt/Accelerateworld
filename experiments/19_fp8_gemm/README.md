# FP8 cuBLASLt GEMM

This experiment studies the first Stage 3 precision path that is intentionally limited to Ada/Blackwell-class RTX hardware.

The comparison keeps the GEMM shape, TN layout, FP32 accumulation/output, workspace budget, autotune procedure and correctness oracle fixed while changing the input representation:

```text
BF16 reference
      ↓
FP8 E4M3
      ↓
FP8 E5M2
```

## Capability boundary

CUDA documents FP8 Tensor Core input support beginning with compute capability 8.9. GPU Baseline v2 therefore records FP8 workloads as `skipped` on RTX 20/Turing and RTX 30/Ampere, while RTX 40/Ada and RTX 50/Blackwell execute them.

The executable repeats the physical compute-capability check at runtime so manually invoking an FP8 mode on an unsupported GPU produces an explicit skip rather than relying only on the repository registry.

## Why TN layout

CUDA 13 cuBLASLt FP8 kernels on Ada and Blackwell GeForce require the TN form: stored A is `K x M` and is multiplied with `CUBLAS_OP_T`; stored B is `K x N` and uses `CUBLAS_OP_N`.

The BF16 reference deliberately uses the same TN layout so the comparison changes precision/scaling rather than changing the matrix-layout problem at the same time.

## Tensorwide scaling

For each FP8 format and for A/B independently:

```text
amax = max(abs(x))
dequant_scale = amax / fp8_max
quantized = fp8(x / dequant_scale)
```

The FP8 maxima used by the experiment are:

- E4M3: `448`;
- E5M2: `57344`.

cuBLASLt receives the dequantization factors through `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` and `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER` with tensorwide scalar FP32 scaling.

The experiment reports:

- A/B amax;
- A/B dequantization scale;
- clipped element count/rate;
- elements close to the FP8 finite limit;
- max input reconstruction error after quantize/dequantize.

This makes scale quality and saturation behavior visible rather than treating FP8 as an opaque library datatype.

## Correctness versus accuracy

Two independent error measurements are retained.

**Correctness error** compares GPU output with CPU FP64 dot products built from the *quantized-and-reconstructed* A/B values. This catches layout, scale, descriptor and execution mistakes without blaming expected FP8 quantization loss.

**Accuracy error** compares the same GPU output with CPU FP64 dot products from the original FP32 master values. This is the actual BF16/FP8 numerical degradation evidence.

Thirty-two deterministic output coordinates are checked for each mode. The pass/fail threshold applies to correctness error; accuracy degradation is reported as benchmark evidence rather than hidden behind a permissive tolerance.

## Autotune path

BF16, E4M3 and E5M2 each receive their own cuBLASLt search:

```text
precision-specific descriptors
        ↓
workspace budget
        ↓
cublasLtMatmulAlgoGetHeuristic
        ↓
heuristic shortlist
        ↓
warm up every executable candidate
        ↓
CUDA Event timing
        ↓
measured lowest-latency winner
```

The experiment records algo ID, tile, stages, split-K, workspace and waves for the measured winner. An algorithm selected for BF16 is never reused blindly for FP8, and E4M3/E5M2 are tuned independently.

`CUBLASLT_MATMUL_DESC_FAST_ACCUM` is explicitly disabled in this stage so accuracy comparisons are not mixed with an additional accumulation-policy variable.

## GPU Baseline v2

Canonical shape: `1024 x 1024 x 1024`, 16 heuristic candidates, 2 warmups, 10 timed iterations and 32 MiB workspace.

Two canonical records are registered:

```text
fp8_e4m3   requires: cuda + fp8
fp8_e5m2   requires: cuda + fp8
```

Each invocation also runs the same-layout BF16 reference and reports FP8 throughput speedup and normalized-error degradation versus BF16.

## Example

```bash
./build/release/bin/aw_fp8_gemm \
  --m 1024 --n 1024 --k 1024 \
  --heuristics 16 --warmup 2 --iterations 10 \
  --workspace-mb 32 --mode all
```

Modes are `all`, `bf16`, `e4m3`, and `e5m2`.

## Evidence boundary

Hosted CI proves CUDA 13 compilation/linking, descriptor/API compatibility for portable targets and CTest registration. It cannot prove actual FP8 Tensor Core execution, throughput, saturation statistics, selected algorithm IDs, accuracy or Compute Sanitizer results. Those belong in GPU Baseline v2 evidence collected on physical Ada/Blackwell RTX hardware.

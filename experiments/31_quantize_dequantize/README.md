# Experiment 31 — Quantize / Dequantize Kernels

This lab is the bridge from floating-point LLM kernels to **weight-only low-precision inference**. The goal is not to implement GEMM yet. It makes the storage format, scale/zero-point semantics, rounding rule, packing rule, error model and standalone quantize/dequantize data movement explicit enough that experiment 32 can consume the exact same representation inside INT8/INT4 weight-only GEMM.

## 1. Dataflow

```text
FP16 / BF16 tensor
        ↓
scale / zero-point
        ↓
round + saturate
        ↓
INT8 or packed INT4
        ↓
dequantize
        ↓
FP16 / BF16 reconstruction
```

The physical benchmark compares PyTorch reference, Triton and handwritten CUDA. Hosted CI only proves reference semantics and portable compilation.

## 2. Formats

`int8_sym` uses `[-127,127]` with zero-point 0. `int8_asym` uses `[-128,127]` with a learned zero-point. `int4_sym` uses signed `[-8,7]` two's-complement nibbles.

Two INT4 weights share one byte:

```text
column 2j     -> low nibble
column 2j + 1 -> high nibble
```

An odd final column leaves the unused high nibble zero.

## 3. Explicit rounding

All providers use **round-half-away-from-zero** before clamping:

```text
x >= 0: floor(x + 0.5)
x <  0: ceil (x - 0.5)
```

This avoids backend-dependent float-to-integer tie behavior.

## 4. Quantization granularity

For `[rows, cols]` the lab supports per-tensor, per-channel (one qparam per row), and group-wise scaling along the last dimension. Group sizes 32/64/128 are modeled. Group metadata is flattened so the next weight-only GEMM can reuse it without changing storage format.

## 5. Files

```text
experiments/31_quantize_dequantize/
├── README.md
├── lab_config.py
├── codec.py
├── reference.py
├── reference_test.py
├── quant_triton.py
├── quant_extension.cpp
├── quant_kernel.cu
├── benchmark.py
└── setup.py
```

`codec.py` is dependency-free and independently proves the signed INT4 nibble representation. Scale generation lives in `reference.py` and is outside timed kernel regions.

## 6. CUDA/Triton kernels

The handwritten CUDA extension exports `quantize_int8`, `dequantize_int8`, `quantize_int4`, and `dequantize_int4`. FP16/BF16 values are widened to FP32 for scale arithmetic. BF16 runtime is gated to CC 8.0+, while compile guards preserve the common RTX20/30/40/50 fatbin target.

Triton uses the same flattened qparam indexing. INT4 quantization is byte-oriented so packing is inside the kernel rather than a later framework op.

## 7. Reference validation

```bash
cd experiments/31_quantize_dequantize
python reference_test.py
```

The CPU test covers explicit rounding, signed INT4 pack/unpack including odd tails, symmetric/asymmetric INT8, group-wise qparams, and finite reconstruction.

## 8. Build

```bash
cd experiments/31_quantize_dequantize
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;12.0"
python setup.py build_ext --inplace
```

Hosted CI compiles/imports this extension but has no physical GPU.

## 9. Physical benchmark

```bash
python experiments/31_quantize_dequantize/benchmark.py \
  --family validation \
  --dtypes fp16,bf16 \
  --formats int8_sym,int8_asym,int4_sym \
  --granularities per_tensor,per_channel,group \
  --group-size 32 \
  --warmup 5 --iterations 20 \
  --json results/quantize-dequantize-validation.json
```

Each record retains quantize/dequantize latency, logical GB/s, RMSE, max error, normalized max error, saturation rate, quantized bytes, compression ratio, qparam count and provider winners. Qparam construction is excluded from timing.

## 10. Shapes

Validation includes an odd-width tensor and a grouped tensor. Larger families represent a decode weight tile, a 4096×4096 attention projection, and a 4096×11008 MLP projection. Hosted CI does not allocate these large workloads.

## 11. Interpretation

This is a memory/conversion experiment, not a GEMM experiment. It should answer whether INT4 packing overhead offsets bandwidth savings, how group-wise qparam lookup changes throughput, what asymmetric zero-point arithmetic costs, and how error changes with format/granularity.

Do **not** infer Tensor Core or weight-only GEMM speed from these standalone kernels.

## 12. Evidence boundary

Hosted CI can prove format algebra, qparam-count rules, INT4 nibble codec, CPU PyTorch reference semantics, Python syntax, multi-architecture CUDA compilation, and extension exports. It cannot prove Triton JIT/runtime correctness, physical CUDA correctness, actual GB/s, packing throughput, or RTX 5060 winners.

Physical evidence belongs in `results/quantize-dequantize-validation.json`.

## 13. ROADMAP connection

After this experiment the storage representation is stable for the strict next item:

```text
INT8 / INT4 weight-only GEMM
        ↓
load low-bit weight + qparams
        ↓
dequantize inside GEMM data path
```

No GEMM is implemented in experiment 31.

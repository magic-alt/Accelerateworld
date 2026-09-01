# 12 — LLM Kernels

This directory starts the transition from generic GPU programming to kernels that appear directly in transformer inference/training stacks.

## v0: RMSNorm

`rmsnorm.py` implements row-wise RMSNorm in Triton and compares it against an equivalent PyTorch expression.

```bash
python rmsnorm.py --rows 4096 --cols 4096
```

## Planned progression

1. RMSNorm
2. fused SiLU/SwiGLU
3. RoPE
4. online softmax
5. fused attention / FlashAttention-style tiling
6. KV-cache update and paged-cache access
7. FP8/INT8/INT4 quantization kernels
8. dequantize + GEMM fusion
9. grouped/persistent GEMM
10. decode-oriented small-batch kernels

Each kernel must retain a framework reference implementation as a correctness oracle and record latency, throughput, shape, dtype, GPU and software versions.

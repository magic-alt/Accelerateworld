# Experiment 12 — RMSNorm v0

RMSNorm is the first explicitly LLM-oriented kernel in the repository. It combines a row reduction with an elementwise affine operation and therefore connects the earlier reduction lessons to a real transformer primitive.

## Mathematical contract

For each row `x` with hidden size D:

```text
mean_square = sum(x_i^2) / D
inv_rms     = 1 / sqrt(mean_square + eps)
y_i         = x_i * inv_rms * weight_i
```

The Triton kernel loads input and weight values, widens them to FP32, performs the squared-value reduction in FP32, computes `rsqrt`, applies the weight and stores back to the input dtype.

## Triton mapping

One Triton program owns one row. The hidden dimension is padded to `triton.next_power_of_2(cols)` and masked at the tail. v0 limits that program block to 65536 elements and chooses 4 or 8 warps based on block size.

This single-program-per-row design is intentionally introductory. Very wide hidden dimensions may require multi-stage reduction or different tiling in a production kernel.

## Reference and run command

The PyTorch reference keeps the reduction in FP32:

```python
x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight
```

Run:

```bash
python experiments/12_llm_kernels/rmsnorm.py --rows 4096 --cols 4096 --dtype fp16
```

The script supports FP16 and FP32, validates with dtype-specific tolerances and benchmarks the PyTorch expression versus Triton using `triton.testing.do_bench`.

## Performance model

RMSNorm is usually more memory/reduction sensitive than GEMM-like. Fusing the reduction, normalization and weight multiply avoids materializing intermediate squared values, means and normalized tensors. The important lesson is therefore data movement and fusion, not only FLOP/s.

## Limitations and next steps

This v0 does not implement BF16, backward/autograd integration, vectorized persistent scheduling or architecture-specific tuning. Those omissions are deliberate: Stage 5 later adds mixed-precision SwiGLU, RoPE, online softmax and attention while experiment 10 provides the PyTorch custom-op integration techniques needed for production framework boundaries.

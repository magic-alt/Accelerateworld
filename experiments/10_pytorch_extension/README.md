# 10 — PyTorch CUDA Extension

This experiment bridges handwritten CUDA into the PyTorch dispatcher using the official `CUDAExtension` + `TORCH_LIBRARY` pattern.

The operator is an LLM-relevant fused activation:

```text
out = silu(gate) * up
```

It demonstrates:

- PyTorch tensor validation and device guards;
- custom CUDA kernel launch on PyTorch's current CUDA stream;
- dispatcher registration for a CUDA backend;
- correctness comparison against native PyTorch;
- end-to-end CUDA Event benchmarking.

## Build

```bash
python setup.py build_ext --inplace
python benchmark.py
```

The first version is intentionally float32-only. FP16/BF16, autograd registration, FakeTensor/meta kernels and `torch.compile` integration are later milestones.

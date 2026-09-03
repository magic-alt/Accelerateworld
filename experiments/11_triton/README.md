# Experiment 11 — Triton Vector Add

This is the repository's Triton entry point. It deliberately repeats the vector-add problem from experiment 01 so the only new concept is Triton's programming and JIT model rather than a new algorithm.

## Triton program model

`vector_add_kernel` maps one Triton program to a block of 256 logical elements:

```python
offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

Instead of explicitly naming CUDA threads, the kernel expresses vector operations over a block of offsets. Triton's compiler lowers that program to GPU code.

## Wrapper and launch

`triton_add` allocates an output tensor, computes the number of programs with `triton.cdiv`, and launches the JIT kernel. The tail mask permits arbitrary vector lengths.

Run on a physical GPU:

```bash
python -m pip install -r python/requirements-gpu.txt
python experiments/11_triton/vector_add.py --elements 16777216
```

Hosted CI only syntax-checks this file; Triton JIT execution requires a compatible physical GPU runtime.

## Correctness and benchmark

The script compares the output with `x+y` using `torch.testing.assert_close`. `triton.testing.do_bench` measures both PyTorch add and the Triton wrapper. Useful bandwidth is calculated from two input reads plus one output write.

Because `triton_add` allocates a fresh output, the provider boundary is not identical to a pure preallocated-kernel microbenchmark. That is acceptable here because the goal is framework-level comparison; later experiments document allocation boundaries explicitly when they matter.

## What to learn

Compare this source with experiment 01. CUDA exposes thread/block mechanics directly; Triton exposes program instances and blocked tensor operations. Neither abstraction removes the need to reason about coalescing, tile sizes, occupancy or memory traffic. Experiment 23 then applies Triton's autotuning model to GEMM, and Stage 5 uses Triton for fused LLM kernels.

## Common failures

A Python import succeeding does not prove a Triton kernel can JIT for the installed GPU. Check the PyTorch CUDA build, Triton version, NVIDIA driver and detected compute capability. Keep compile/static evidence separate from physical runtime results.

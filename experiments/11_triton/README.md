# 11 — Triton

This stage repeats a familiar workload in Triton so CUDA C++ and compiler-generated GPU kernels can be compared under the same methodology.

`vector_add.py` compares a Triton vector-add kernel with `torch.add` and validates numerical equivalence. It is intentionally simple before moving to auto-tuned GEMM, fused softmax and attention.

```bash
python vector_add.py
```

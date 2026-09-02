# 10 — PyTorch CUDA Extension

This experiment bridges handwritten CUDA into the PyTorch dispatcher using the official `CUDAExtension` + `TORCH_LIBRARY` pattern, then layers autograd and compiler metadata registrations on top from Python.

The operator is an LLM-relevant fused activation:

```text
out = silu(gate) * up
```

It now demonstrates the complete framework-integration path for one opaque CUDA operator:

- PyTorch tensor validation and device guards;
- launch on PyTorch's current CUDA stream;
- `TORCH_LIBRARY` schema + CUDA dispatch registration;
- `torch.library.register_autograd` training semantics;
- `torch.library.register_fake` FakeTensor/meta semantics;
- full `torch.library.opcheck` coverage;
- Dynamo full-graph capture with no graph break;
- AOTAutograd forward/backward tracing;
- TorchInductor execution around an opaque custom-op boundary;
- forced dynamic-shape stress testing;
- eager-vs-compiled CUDA Event benchmarking.

## Registration stack

The compiled extension owns the real CUDA execution path:

```text
extension.cpp
    TORCH_LIBRARY(accelerateworld)
        ↓
accelerateworld::silu_mul schema
        ↓
TORCH_LIBRARY_IMPL(..., CUDA)
        ↓
silu_mul_cuda
        ↓
SiluMulKernel
```

`accelerateworld_ops.py` loads that schema first and then adds the higher-level contracts:

```text
accelerateworld_cuda import
        ↓
register_fake
        ├── shape
        ├── strides
        ├── dtype
        ├── layout
        └── logical CUDA device
        ↓
register_autograd
        ↓
setup_context + backward
        ↓
ordinary PyTorch ops
```

The FakeTensor implementation does not inspect storage or data. The real CUDA implementation returns `at::empty_like(gate)`, so the fake implementation returns `torch.empty_like(gate)` after reproducing the CUDA kernel's input contract: float32, equal shapes and contiguous CUDA tensors.

For

```text
y = silu(gate) * up
```

with `s = sigmoid(gate)`, the registered gradients are:

```text
dy/dgate = up * s * (1 + gate * (1 - s))
dy/dup   = silu(gate)
```

The backward remains composed only of PyTorch-understood operators. The forward custom CUDA operator therefore stays opaque, while AOTAutograd and Inductor can still trace and optimize the surrounding graph and the registered backward formula.

## Build

```bash
python setup.py build_ext --inplace
```

Callers should import the Python registration shim rather than importing the binary directly:

```python
from accelerateworld_ops import silu_mul
```

## Forward and autograd validation

```bash
python benchmark.py
python autograd_test.py --elements 4096
```

`autograd_test.py` compares the custom CUDA path against native `F.silu(gate) * up`, including both one-sided `needs_input_grad` cases.

The v0 CUDA kernel remains float32-only, so double-precision `gradcheck` is intentionally not used yet. Numerical gradient correctness is compared directly against native PyTorch autograd on identical float32 inputs.

## FakeTensor metadata validation

Hosted CI can validate the fake kernel without a physical GPU because FakeTensor carries a logical CUDA device without allocating CUDA storage:

```bash
python fake_tensor_test.py
```

The test requires:

- a Meta dispatch registration from `register_fake`;
- FakeTensor output rather than a real allocation;
- exact shape propagation;
- exact stride propagation;
- float32 dtype propagation;
- logical CUDA device propagation;
- layout propagation;
- rejection of mismatched input shapes.

## Full `opcheck`

The compiler validation runs the default PyTorch `torch.library.opcheck` suite on real CUDA tensors:

```text
test_schema
        ↓
test_autograd_registration
        ↓
test_faketensor
        ↓
test_aot_dispatch_dynamic
```

`opcheck` validates registration contracts rather than mathematical accuracy, so the repository retains separate eager/native forward and gradient comparisons.

## `torch.compile` validation

On a physical CUDA GPU:

```bash
python compile_test.py --rows 16 --cols 257
```

The test deliberately separates compiler layers:

```text
custom Python wrapper
        ↓
Dynamo / fullgraph=True
        ↓
FX graph contains accelerateworld::silu_mul
        ↓
AOTAutograd
        ↓
registered backward graph
        ↓
TorchInductor
        ↓
compiled surrounding kernels
        ↓
opaque custom CUDA op boundary
```

Three backends are exercised in sequence:

```text
backend="eager"      → Dynamo capture only
backend="aot_eager"  → Dynamo + AOTAutograd
backend="inductor"   → Dynamo + AOTAutograd + TorchInductor
```

All three must match the native PyTorch forward/loss/gradient result. `fullgraph=True` makes any graph break a hard failure, and `torch._dynamo.explain` separately verifies one graph, zero graph breaks, and retention of the custom operator in the captured FX graph.

### Dynamic shapes

The lab also forces `dynamic=True` as a stress test and invokes the same compiled callable with:

```text
(2, 257)
    ↓
(4, 513)
    ↓
(8, 1025)
```

A counting backend requires this sequence to produce one Dynamo graph. This is an experiment-level stress test, not a blanket production recommendation to force every dimension dynamic; real applications should use automatic or explicitly annotated dynamic dimensions where appropriate.

## Eager vs compile benchmark

```bash
python compile_benchmark.py --elements 4194304 --warmup 10 --iterations 30
```

The benchmark uses a small graph around the custom boundary:

```text
silu_mul custom CUDA op
        ↓
tanh
        ↓
scale + add
        ↓
loss for training path
```

It records:

- native PyTorch eager forward latency;
- custom-op eager forward latency;
- custom-op Inductor forward latency;
- eager vs compiled forward speedup;
- custom-op eager forward+backward latency;
- custom-op compiled forward+backward latency;
- eager vs compiled training speedup.

Compilation happens before the timed region. Runtime measurements use CUDA Events, so first-compile latency is deliberately excluded from steady-state execution results.

The expected lesson is not that `torch.compile` removes or fuses through an opaque custom CUDA operator. It cannot. The custom operator remains a framework-visible boundary; Inductor can optimize compatible operations around that boundary, while AOTAutograd can compile the PyTorch-composed backward graph.

## Evidence boundary

Hosted `python-extension-build` CI proves:

- PyTorch 2.13 / CUDA 13 extension compilation for RTX 20/30/40/50 targets;
- CUDA dispatcher registration;
- Autograd dispatcher registration;
- Meta/FakeTensor registration;
- storage-free FakeTensor metadata propagation;
- analytical backward formula correctness on ordinary PyTorch tensors.

The self-hosted GPU workflow is responsible for evidence that requires real CUDA execution:

- full `opcheck`, including `test_faketensor` and `test_aot_dispatch_dynamic` against the real CUDA kernel;
- Dynamo no-graph-break capture;
- dynamic-shape execution;
- AOTAutograd forward/backward execution;
- TorchInductor forward/backward execution;
- eager-vs-compiled runtime measurements.

Hosted compilation must not be presented as physical-GPU compiler performance evidence.

## Next ROADMAP item

After this milestone, the next unfinished Stage 4 item is **Triton auto-tuned GEMM**.

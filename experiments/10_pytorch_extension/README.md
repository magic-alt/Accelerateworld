# 10 — PyTorch CUDA Extension

This experiment bridges handwritten CUDA into the PyTorch dispatcher using the official `CUDAExtension` + `TORCH_LIBRARY` pattern, then extends the operator with Python-side `torch.library` registrations.

The operator is an LLM-relevant fused activation:

```text
out = silu(gate) * up
```

It demonstrates:

- PyTorch tensor validation and device guards;
- custom CUDA kernel launch on PyTorch's current CUDA stream;
- dispatcher registration for a CUDA backend;
- Python-side `torch.library.register_autograd` integration;
- correctness comparison against native PyTorch;
- end-to-end CUDA Event benchmarking.

## Registration layers

The compiled extension owns the opaque CUDA operation:

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

`accelerateworld_ops.py` must load that compiled registration first, then adds the training contract:

```text
accelerateworld_cuda import
        ↓
torch.library.register_autograd
        ↓
setup_context saves gate/up
        ↓
backward uses ordinary PyTorch ops
```

For

```text
y = silu(gate) * up
```

with `s = sigmoid(gate)`, the registered gradients are:

```text
dy/dgate = up * s * (1 + gate * (1 - s))
dy/dup   = silu(gate)
```

The backward is intentionally expressed only with PyTorch-understood operators. No CUDA pointer access or custom backward kernel is hidden inside the autograd formula. This follows PyTorch's custom-operator integration model and leaves the backward graph available to later compiler transformations.

## Build and forward benchmark

```bash
python setup.py build_ext --inplace
python benchmark.py
```

`setup.py` installs the `accelerateworld_ops` Python shim together with the compiled extension, so callers should import the shim rather than importing the binary module directly:

```python
from accelerateworld_ops import silu_mul
```

## Autograd validation

On a CUDA GPU:

```bash
python autograd_test.py --elements 4096
```

The validation compares the custom operator against native:

```python
F.silu(gate) * up
```

and checks:

- forward output;
- gradient with respect to `gate`;
- gradient with respect to `up`;
- `needs_input_grad` behavior when only one input requires gradients;
- `torch.library.opcheck` schema and explicit autograd-registration checks.

Standard double-precision `torch.autograd.gradcheck` is not used yet because the underlying v0 CUDA kernel intentionally supports float32 only. Numerical gradient correctness is instead checked directly against native PyTorch's autograd result on the same float32 inputs. Extending the operator's dtype surface belongs to the later mixed-precision LLM-kernel work.

## CI boundary

Hosted extension-build CI can prove that:

- the CUDA extension compiles for the configured RTX architecture set;
- the Python registration shim imports after the compiled operator is loaded;
- an explicit Autograd dispatcher kernel is registered;
- the analytical backward formula agrees with native PyTorch on CPU tensors.

The self-hosted GPU validation additionally executes the real CUDA forward/backward path and `opcheck` against CUDA tensors.

## Next compiler integration step

The next ROADMAP item is **FakeTensor/meta kernel and `torch.compile`**. The current autograd registration is deliberately kept separate from that work: this PR establishes training semantics first, while the next PR will define output metadata for storage-less FakeTensors and validate eager-vs-compiled execution.

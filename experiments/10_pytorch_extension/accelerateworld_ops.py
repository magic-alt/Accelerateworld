"""Python-side registrations for Accelerateworld custom PyTorch operators.

The compiled ``accelerateworld_cuda`` module owns the operator schema and CUDA
implementation. Registrations that integrate that opaque CUDA operator with
higher-level PyTorch subsystems live here so they can be expressed as ordinary
PyTorch operations.
"""

from __future__ import annotations

import torch

import accelerateworld_cuda  # noqa: F401  # loads TORCH_LIBRARY registrations first


@torch.library.register_fake("accelerateworld::silu_mul")
def _silu_mul_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Describe the CUDA operator's output metadata without touching storage."""

    torch._check(gate.device.type == "cuda", lambda: "gate must be a CUDA tensor")
    torch._check(up.device.type == "cuda", lambda: "up must be a CUDA tensor")
    torch._check(gate.dtype == torch.float32, lambda: "gate must be float32")
    torch._check(up.dtype == torch.float32, lambda: "up must be float32")
    torch._check(gate.dim() == up.dim(), lambda: "gate and up must have the same rank")
    for dim in range(gate.dim()):
        torch._check(
            gate.size(dim) == up.size(dim),
            lambda: "gate and up must have the same shape",
        )
    torch._check(gate.is_contiguous(), lambda: "gate must be contiguous")
    torch._check(up.is_contiguous(), lambda: "up must be contiguous")

    # silu_mul_cuda allocates with at::empty_like(gate), so the fake result must
    # preserve the same shape, strides, dtype, layout and logical CUDA device.
    return torch.empty_like(gate)


def _silu_mul_backward_formula(
    gate: torch.Tensor, up: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return analytical gradients using only PyTorch-understood operations."""

    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    grad_gate = grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_up = grad_output * silu
    return grad_gate, grad_up


def _setup_context(ctx: object, inputs: tuple[torch.Tensor, torch.Tensor], output: torch.Tensor) -> None:
    del output
    gate, up = inputs
    ctx.save_for_backward(gate, up)


def _backward(
    ctx: object, grad_output: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    gate, up = ctx.saved_tensors
    grad_gate, grad_up = _silu_mul_backward_formula(gate, up, grad_output)
    if not ctx.needs_input_grad[0]:
        grad_gate = None
    if not ctx.needs_input_grad[1]:
        grad_up = None
    return grad_gate, grad_up


torch.library.register_autograd(
    "accelerateworld::silu_mul",
    _backward,
    setup_context=_setup_context,
)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up`` CUDA op with autograd and compiler metadata."""

    return torch.ops.accelerateworld.silu_mul(gate, up)

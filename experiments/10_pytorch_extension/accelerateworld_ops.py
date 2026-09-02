"""Python-side registrations for Accelerateworld custom PyTorch operators.

The compiled ``accelerateworld_cuda`` module owns the operator schema and CUDA
implementation.  Registrations that integrate that opaque CUDA operator with
higher-level PyTorch subsystems live here so they can be expressed as ordinary
PyTorch operations.
"""

from __future__ import annotations

import torch

import accelerateworld_cuda  # noqa: F401  # loads TORCH_LIBRARY registrations first


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
    """Fused ``silu(gate) * up`` custom CUDA operator with autograd support."""

    return torch.ops.accelerateworld.silu_mul(gate, up)

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from accelerateworld_ops import silu_mul


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left - right).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4096)
    args = parser.parse_args()

    if args.elements <= 0:
        raise ValueError("--elements must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    gate_data = torch.randn(args.elements, device=device, dtype=torch.float32) * 0.75
    up_data = torch.randn_like(gate_data)
    grad_output = torch.randn_like(gate_data)

    gate_ref = gate_data.detach().clone().requires_grad_(True)
    up_ref = up_data.detach().clone().requires_grad_(True)
    reference = F.silu(gate_ref) * up_ref
    reference.backward(grad_output)

    gate_custom = gate_data.detach().clone().requires_grad_(True)
    up_custom = up_data.detach().clone().requires_grad_(True)
    custom = silu_mul(gate_custom, up_custom)
    custom.backward(grad_output)

    forward_error = _max_abs(custom.detach(), reference.detach())
    gate_grad_error = _max_abs(gate_custom.grad, gate_ref.grad)
    up_grad_error = _max_abs(up_custom.grad, up_ref.grad)

    torch.testing.assert_close(custom.detach(), reference.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gate_custom.grad, gate_ref.grad, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(up_custom.grad, up_ref.grad, rtol=2e-5, atol=2e-5)

    # Exercise both needs_input_grad branches independently.
    gate_only = gate_data.detach().clone().requires_grad_(True)
    up_constant = up_data.detach().clone()
    gate_only_grad = torch.autograd.grad(silu_mul(gate_only, up_constant).sum(), gate_only)[0]
    gate_only_ref = torch.autograd.grad((F.silu(gate_only) * up_constant).sum(), gate_only)[0]
    torch.testing.assert_close(gate_only_grad, gate_only_ref, rtol=3e-5, atol=3e-5)

    gate_constant = gate_data.detach().clone()
    up_only = up_data.detach().clone().requires_grad_(True)
    up_only_grad = torch.autograd.grad(silu_mul(gate_constant, up_only).sum(), up_only)[0]
    up_only_ref = torch.autograd.grad((F.silu(gate_constant) * up_only).sum(), up_only)[0]
    torch.testing.assert_close(up_only_grad, up_only_ref, rtol=2e-5, atol=2e-5)

    opcheck = torch.library.opcheck(
        torch.ops.accelerateworld.silu_mul.default,
        (
            gate_data.detach().clone().requires_grad_(True),
            up_data.detach().clone().requires_grad_(True),
        ),
        test_utils=("test_schema", "test_autograd_registration"),
    )

    print("PyTorch autograd registration — fused SiLU*mul")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Elements: {args.elements}")
    print(f"  Forward max abs error: {forward_error:.6g}")
    print(f"  gate gradient max abs error: {gate_grad_error:.6g}")
    print(f"  up gradient max abs error: {up_grad_error:.6g}")
    print(f"  opcheck: {opcheck}")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()

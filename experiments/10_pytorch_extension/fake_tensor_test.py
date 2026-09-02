from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from accelerateworld_ops import silu_mul


def main() -> None:
    op_name = "accelerateworld::silu_mul"
    if not torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta"):
        raise RuntimeError("FakeTensor registration did not install a Meta kernel")

    mode = FakeTensorMode()
    with mode:
        gate = torch.empty((7, 11), device="cuda", dtype=torch.float32)
        up = torch.empty_like(gate)
        output = silu_mul(gate, up)

        if not isinstance(output, FakeTensor):
            raise RuntimeError(f"expected FakeTensor output, got {type(output)!r}")
        if output.shape != gate.shape:
            raise RuntimeError(f"shape mismatch: {output.shape} != {gate.shape}")
        if output.stride() != gate.stride():
            raise RuntimeError(f"stride mismatch: {output.stride()} != {gate.stride()}")
        if output.dtype != gate.dtype:
            raise RuntimeError(f"dtype mismatch: {output.dtype} != {gate.dtype}")
        if output.device != gate.device:
            raise RuntimeError(f"device mismatch: {output.device} != {gate.device}")
        if output.layout != gate.layout:
            raise RuntimeError(f"layout mismatch: {output.layout} != {gate.layout}")

        bad_up = torch.empty((7, 13), device="cuda", dtype=torch.float32)
        try:
            silu_mul(gate, bad_up)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("FakeTensor kernel accepted mismatched input shapes")

    print("PyTorch FakeTensor metadata — fused SiLU*mul")
    print("  Meta dispatch registration: PASS")
    print("  Logical device: cuda")
    print("  Shape/stride/dtype/device/layout propagation: PASS")
    print("  Contract validation: PASS")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()

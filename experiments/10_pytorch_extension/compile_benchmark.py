from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from accelerateworld_ops import silu_mul


def custom_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    fused = silu_mul(gate, up)
    return torch.tanh(fused) + 0.125 * fused


def native_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    fused = F.silu(gate) * up
    return torch.tanh(fused) + 0.125 * fused


def custom_loss(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return custom_forward(gate, up).square().mean()


def _bench(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if args.elements <= 0:
        raise ValueError("--elements must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    torch.manual_seed(2026)
    gate = (torch.randn(args.elements, device="cuda", dtype=torch.float32) * 0.75).contiguous()
    up = torch.randn_like(gate)

    compiled_forward = torch.compile(
        custom_forward,
        backend="inductor",
        fullgraph=True,
    )
    compiled_loss = torch.compile(
        custom_loss,
        backend="inductor",
        fullgraph=True,
    )

    # Compile before entering the timed regions.
    eager_out = custom_forward(gate, up)
    compiled_out = compiled_forward(gate, up)
    native_out = native_forward(gate, up)
    torch.testing.assert_close(eager_out, native_out, rtol=5e-5, atol=2e-5)
    torch.testing.assert_close(compiled_out, native_out, rtol=5e-5, atol=2e-5)

    gate_eager = gate.detach().clone().requires_grad_(True)
    up_eager = up.detach().clone().requires_grad_(True)
    gate_compiled = gate.detach().clone().requires_grad_(True)
    up_compiled = up.detach().clone().requires_grad_(True)

    # Force AOTAutograd/Inductor backward compilation before timing.
    compiled_loss(gate_compiled, up_compiled).backward()
    gate_compiled.grad = None
    up_compiled.grad = None
    torch.cuda.synchronize()

    native_forward_ms = _bench(
        lambda: native_forward(gate, up), args.warmup, args.iterations
    )
    eager_forward_ms = _bench(
        lambda: custom_forward(gate, up), args.warmup, args.iterations
    )
    compiled_forward_ms = _bench(
        lambda: compiled_forward(gate, up), args.warmup, args.iterations
    )

    def eager_train_step() -> None:
        gate_eager.grad = None
        up_eager.grad = None
        custom_loss(gate_eager, up_eager).backward()

    def compiled_train_step() -> None:
        gate_compiled.grad = None
        up_compiled.grad = None
        compiled_loss(gate_compiled, up_compiled).backward()

    eager_train_ms = _bench(eager_train_step, args.warmup, args.iterations)
    compiled_train_ms = _bench(compiled_train_step, args.warmup, args.iterations)

    print("PyTorch torch.compile benchmark — fused SiLU*mul boundary")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Elements: {args.elements}")
    print(f"  Native eager forward: {native_forward_ms:.4f} ms")
    print(f"  Custom eager forward: {eager_forward_ms:.4f} ms")
    print(f"  Custom compiled forward: {compiled_forward_ms:.4f} ms")
    print(f"  Forward compile speedup: {eager_forward_ms / compiled_forward_ms:.3f}x")
    print(f"  Custom eager forward+backward: {eager_train_ms:.4f} ms")
    print(f"  Custom compiled forward+backward: {compiled_train_ms:.4f} ms")
    print(f"  Training compile speedup: {eager_train_ms / compiled_train_ms:.3f}x")
    print("  Timing excludes first compilation and uses CUDA Events")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()

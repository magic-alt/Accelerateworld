import argparse
import statistics

import torch
import torch.nn.functional as F

from accelerateworld_ops import silu_mul


def bench(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
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
    parser.add_argument("--elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    gate = torch.randn(args.elements, device="cuda", dtype=torch.float32)
    up = torch.randn_like(gate)

    reference = F.silu(gate) * up
    custom = silu_mul(gate, up)
    max_error = (reference - custom).abs().max().item()
    if max_error > 2e-5:
        raise RuntimeError(f"correctness check failed: max_error={max_error}")

    native_ms = bench(lambda: F.silu(gate) * up, args.warmup, args.iterations)
    custom_ms = bench(lambda: silu_mul(gate, up), args.warmup, args.iterations)

    print("PyTorch CUDA Extension — fused SiLU*mul")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Elements: {args.elements}")
    print(f"  Native PyTorch: {native_ms:.4f} ms")
    print(f"  Custom CUDA op: {custom_ms:.4f} ms")
    print(f"  Speedup: {native_ms / custom_ms:.3f}x")
    print(f"  Max abs error: {max_error:.6g}")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, 256),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    x = torch.randn(args.elements, device="cuda")
    y = torch.randn_like(x)
    reference = x + y
    actual = triton_add(x, y)
    torch.testing.assert_close(reference, actual)

    torch_ms = triton.testing.do_bench(lambda: x + y)
    triton_ms = triton.testing.do_bench(lambda: triton_add(x, y))
    bytes_moved = 3 * x.numel() * x.element_size()
    bandwidth = bytes_moved / (triton_ms / 1000.0) / 1e9

    print("Triton vector add")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Elements: {args.elements}")
    print(f"  torch.add: {torch_ms:.4f} ms")
    print(f"  Triton: {triton_ms:.4f} ms")
    print(f"  Triton useful bandwidth: {bandwidth:.2f} GB/s")
    print(f"  Speedup: {torch_ms / triton_ms:.3f}x")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()

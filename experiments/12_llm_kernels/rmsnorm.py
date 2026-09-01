import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_row,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = x * inv_rms * weight
    tl.store(out_ptr + row * stride_row + offsets, output, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("v0 expects a 2D tensor [rows, hidden_size]")
    rows, cols = x.shape
    block_size = triton.next_power_of_2(cols)
    if block_size > 65536:
        raise ValueError("hidden size too large for this introductory single-program RMSNorm")
    out = torch.empty_like(x)
    rmsnorm_kernel[(rows,)](
        x,
        weight,
        out,
        x.stride(0),
        n_cols=cols,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 4096 else 4,
    )
    return out


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=dtype)
    weight = torch.randn((args.cols,), device="cuda", dtype=dtype)

    reference = torch_rmsnorm(x, weight, args.eps).to(dtype)
    actual = triton_rmsnorm(x, weight, args.eps)
    torch.testing.assert_close(reference, actual, rtol=2e-2 if dtype == torch.float16 else 2e-4,
                               atol=2e-2 if dtype == torch.float16 else 2e-4)

    torch_ms = triton.testing.do_bench(lambda: torch_rmsnorm(x, weight, args.eps))
    triton_ms = triton.testing.do_bench(lambda: triton_rmsnorm(x, weight, args.eps))

    print("LLM kernel — RMSNorm")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Shape: {tuple(x.shape)}, dtype={dtype}")
    print(f"  PyTorch expression: {torch_ms:.4f} ms")
    print(f"  Triton RMSNorm: {triton_ms:.4f} ms")
    print(f"  Speedup: {torch_ms / triton_ms:.3f}x")
    print("  Validation: PASS")


if __name__ == "__main__":
    main()

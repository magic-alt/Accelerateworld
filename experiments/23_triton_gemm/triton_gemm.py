from __future__ import annotations

import torch
import triton
import triton.language as tl

from lab_config import AUTOTUNE_CONFIG_SPECS


def _triton_configs() -> list[triton.Config]:
    configs: list[triton.Config] = []
    for spec in AUTOTUNE_CONFIG_SPECS:
        kwargs = {
            key: value
            for key, value in spec.items()
            if key not in {"num_warps", "num_stages"}
        }
        configs.append(
            triton.Config(
                kwargs,
                num_warps=spec["num_warps"],
                num_stages=spec["num_stages"],
            )
        )
    return configs


@triton.autotune(
    configs=_triton_configs(),
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k_block * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_offset < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] + k_offset < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> tuple[int, int, int]:
    if not (a.is_cuda and b.is_cuda and out.is_cuda):
        raise ValueError("Triton GEMM requires CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16 or out.dtype != torch.float16:
        raise ValueError("the Stage 4 Triton GEMM lab is intentionally FP16-only")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("a, b and out must be rank-2 matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible GEMM shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        raise ValueError("the Stage 4 Triton GEMM lab requires contiguous row-major tensors")
    return int(m), int(n), int(k)


def matmul_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Run the autotuned FP16 GEMM into a preallocated FP16 output tensor."""

    m, n, k = _validate_inputs(a, b, out)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    _matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2 matrices")
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    return matmul_into(a, b, out)


def best_config_dict() -> dict[str, int] | None:
    """Return the most recently selected autotune config after a kernel invocation."""

    config = getattr(_matmul_kernel, "best_config", None)
    if config is None:
        return None
    result = dict(config.kwargs)
    result["num_warps"] = int(config.num_warps)
    result["num_stages"] = int(config.num_stages)
    return {key: int(value) for key, value in result.items()}

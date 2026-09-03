from __future__ import annotations

import torch
import triton
import triton.language as tl

from lab_config import LAYOUT_IDS
from reference import capacity_from_cache, validate_positions


@triton.jit
def _kv_update_kernel(
    cache_k,
    cache_v,
    new_k,
    new_v,
    positions,
    batch: tl.constexpr,
    kv_heads: tl.constexpr,
    tokens: tl.constexpr,
    capacity: tl.constexpr,
    head_dim: tl.constexpr,
    layout: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    token = row % tokens
    tmp = row // tokens
    head = tmp % kv_heads
    batch_index = tmp // kv_heads
    position = tl.load(positions + batch_index * tokens + token)

    d = tl.arange(0, block_d)
    mask = d < head_dim
    source = ((batch_index * kv_heads + head) * tokens + token) * head_dim + d
    if layout == 0:
        target = ((batch_index * capacity + position) * kv_heads + head) * head_dim + d
    else:
        target = ((batch_index * kv_heads + head) * capacity + position) * head_dim + d

    k = tl.load(new_k + source, mask=mask)
    v = tl.load(new_v + source, mask=mask)
    tl.store(cache_k + target, k, mask=mask)
    tl.store(cache_v + target, v, mask=mask)


@triton.jit
def _kv_read_kernel(
    cache_k,
    cache_v,
    positions,
    out_k,
    out_v,
    batch: tl.constexpr,
    kv_heads: tl.constexpr,
    tokens: tl.constexpr,
    capacity: tl.constexpr,
    head_dim: tl.constexpr,
    layout: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    token = row % tokens
    tmp = row // tokens
    head = tmp % kv_heads
    batch_index = tmp // kv_heads
    position = tl.load(positions + batch_index * tokens + token)

    d = tl.arange(0, block_d)
    mask = d < head_dim
    if layout == 0:
        source = ((batch_index * capacity + position) * kv_heads + head) * head_dim + d
    else:
        source = ((batch_index * kv_heads + head) * capacity + position) * head_dim + d
    target = ((batch_index * kv_heads + head) * tokens + token) * head_dim + d

    k = tl.load(cache_k + source, mask=mask)
    v = tl.load(cache_v + source, mask=mask)
    tl.store(out_k + target, k, mask=mask)
    tl.store(out_v + target, v, mask=mask)


def _validate_common(cache_k: torch.Tensor, cache_v: torch.Tensor, positions: torch.Tensor, layout: str, check_bounds: bool) -> int:
    if layout not in LAYOUT_IDS:
        raise KeyError(f"unsupported layout: {layout}")
    if not (cache_k.is_cuda and cache_v.is_cuda and positions.is_cuda):
        raise ValueError("Triton KV-cache tensors must be CUDA tensors")
    if cache_k.shape != cache_v.shape or cache_k.ndim != 4:
        raise ValueError("cache K/V must be matching rank-4 tensors")
    if positions.ndim != 2 or positions.dtype != torch.int64:
        raise ValueError("positions must be CUDA int64 [batch, tokens]")
    if not (cache_k.is_contiguous() and cache_v.is_contiguous() and positions.is_contiguous()):
        raise ValueError("cache and positions must be contiguous")
    capacity = capacity_from_cache(cache_k, layout)
    if check_bounds:
        validate_positions(positions, capacity)
    return capacity


def kv_cache_update(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    *,
    layout: str,
    check_bounds: bool = True,
) -> None:
    capacity = _validate_common(cache_k, cache_v, positions, layout, check_bounds)
    if new_k.shape != new_v.shape or new_k.ndim != 4:
        raise ValueError("new K/V must be matching [batch, kv_heads, tokens, head_dim]")
    batch, kv_heads, tokens, head_dim = map(int, new_k.shape)
    if positions.shape != (batch, tokens):
        raise ValueError("positions must match [batch, tokens]")
    if head_dim not in (64, 128):
        raise ValueError("v0 supports head_dim 64 or 128")
    if not (new_k.is_contiguous() and new_v.is_contiguous()):
        raise ValueError("new K/V must be contiguous")
    block_d = triton.next_power_of_2(head_dim)
    _kv_update_kernel[(batch * kv_heads * tokens,)](
        cache_k, cache_v, new_k, new_v, positions,
        batch=batch, kv_heads=kv_heads, tokens=tokens, capacity=capacity,
        head_dim=head_dim, layout=LAYOUT_IDS[layout], block_d=block_d,
        num_warps=4,
    )


def kv_cache_read(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    positions: torch.Tensor,
    *,
    layout: str,
    check_bounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    capacity = _validate_common(cache_k, cache_v, positions, layout, check_bounds)
    batch, tokens = map(int, positions.shape)
    if layout == "token_major":
        kv_heads, head_dim = int(cache_k.shape[2]), int(cache_k.shape[3])
    else:
        kv_heads, head_dim = int(cache_k.shape[1]), int(cache_k.shape[3])
    out_k = torch.empty((batch, kv_heads, tokens, head_dim), device=cache_k.device, dtype=cache_k.dtype)
    out_v = torch.empty_like(out_k)
    block_d = triton.next_power_of_2(head_dim)
    _kv_read_kernel[(batch * kv_heads * tokens,)](
        cache_k, cache_v, positions, out_k, out_v,
        batch=batch, kv_heads=kv_heads, tokens=tokens, capacity=capacity,
        head_dim=head_dim, layout=LAYOUT_IDS[layout], block_d=block_d,
        num_warps=4,
    )
    return out_k, out_v

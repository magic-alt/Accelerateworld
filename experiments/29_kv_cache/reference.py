from __future__ import annotations

import torch

from lab_config import LAYOUTS


def cache_shape(batch: int, kv_heads: int, capacity: int, head_dim: int, layout: str) -> tuple[int, ...]:
    if layout == "token_major":
        return (batch, capacity, kv_heads, head_dim)
    if layout == "head_major":
        return (batch, kv_heads, capacity, head_dim)
    raise KeyError(f"unsupported layout: {layout}")


def allocate_cache(
    batch: int,
    kv_heads: int,
    capacity: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = cache_shape(batch, kv_heads, capacity, head_dim, layout)
    return (
        torch.zeros(shape, dtype=dtype, device=device),
        torch.zeros(shape, dtype=dtype, device=device),
    )


def capacity_from_cache(cache: torch.Tensor, layout: str) -> int:
    if layout == "token_major":
        return int(cache.shape[1])
    if layout == "head_major":
        return int(cache.shape[2])
    raise KeyError(f"unsupported layout: {layout}")


def validate_positions(positions: torch.Tensor, capacity: int) -> None:
    if positions.ndim != 2:
        raise ValueError("positions must be [batch, tokens]")
    if positions.dtype != torch.int64:
        raise TypeError("positions must use torch.int64")
    if positions.numel() == 0:
        raise ValueError("positions must not be empty")
    minimum = int(positions.min().item())
    maximum = int(positions.max().item())
    if minimum < 0 or maximum >= capacity:
        raise IndexError(f"KV-cache position range [{minimum}, {maximum}] exceeds capacity {capacity}")


def validate_contract(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    layout: str,
) -> tuple[int, int, int, int, int]:
    if layout not in LAYOUTS:
        raise KeyError(f"unsupported layout: {layout}")
    if cache_k.shape != cache_v.shape or new_k.shape != new_v.shape:
        raise ValueError("K/V tensor pairs must have matching shapes")
    if new_k.ndim != 4:
        raise ValueError("new K/V must be [batch, kv_heads, tokens, head_dim]")
    batch, kv_heads, tokens, head_dim = map(int, new_k.shape)
    if positions.shape != (batch, tokens):
        raise ValueError("positions shape must match [batch, tokens]")
    expected_cache_rank = 4
    if cache_k.ndim != expected_cache_rank:
        raise ValueError("cache tensors must be rank-4")
    capacity = capacity_from_cache(cache_k, layout)
    expected = cache_shape(batch, kv_heads, capacity, head_dim, layout)
    if tuple(cache_k.shape) != expected:
        raise ValueError(f"cache shape {tuple(cache_k.shape)} does not match {expected}")
    if cache_k.dtype != new_k.dtype or cache_v.dtype != new_v.dtype:
        raise TypeError("cache and update dtypes must match")
    validate_positions(positions, capacity)
    return batch, kv_heads, tokens, head_dim, capacity


def kv_cache_update_reference(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    *,
    layout: str,
    check_bounds: bool = True,
) -> None:
    batch, kv_heads, tokens, head_dim = map(int, new_k.shape)
    capacity = capacity_from_cache(cache_k, layout)
    if check_bounds:
        validate_contract(cache_k, cache_v, new_k, new_v, positions, layout)

    if layout == "head_major":
        index = positions[:, None, :, None].expand(batch, kv_heads, tokens, head_dim)
        cache_k.scatter_(2, index, new_k)
        cache_v.scatter_(2, index, new_v)
        return

    source_k = new_k.permute(0, 2, 1, 3)
    source_v = new_v.permute(0, 2, 1, 3)
    index = positions[:, :, None, None].expand(batch, tokens, kv_heads, head_dim)
    cache_k.scatter_(1, index, source_k)
    cache_v.scatter_(1, index, source_v)


def kv_cache_read_reference(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    positions: torch.Tensor,
    *,
    layout: str,
    check_bounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    capacity = capacity_from_cache(cache_k, layout)
    if check_bounds:
        validate_positions(positions, capacity)
    batch, tokens = map(int, positions.shape)

    if layout == "head_major":
        kv_heads = int(cache_k.shape[1])
        head_dim = int(cache_k.shape[3])
        index = positions[:, None, :, None].expand(batch, kv_heads, tokens, head_dim)
        return cache_k.gather(2, index), cache_v.gather(2, index)

    kv_heads = int(cache_k.shape[2])
    head_dim = int(cache_k.shape[3])
    index = positions[:, None, :, None].expand(batch, kv_heads, tokens, head_dim)
    k = cache_k.permute(0, 2, 1, 3).gather(2, index)
    v = cache_v.permute(0, 2, 1, 3).gather(2, index)
    return k.contiguous(), v.contiguous()

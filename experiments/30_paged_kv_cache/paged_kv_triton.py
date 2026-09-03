from __future__ import annotations

import torch
import triton
import triton.language as tl

from reference import validate_block_table, validate_positions


@triton.jit
def _paged_kv_update_kernel(
    page_k,
    page_v,
    new_k,
    new_v,
    positions,
    block_table,
    batch: tl.constexpr,
    kv_heads: tl.constexpr,
    tokens: tl.constexpr,
    num_pages: tl.constexpr,
    max_blocks: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    token = row % tokens
    tmp = row // tokens
    head = tmp % kv_heads
    batch_index = tmp // kv_heads

    position = tl.load(positions + batch_index * tokens + token)
    logical_block = position // page_size
    slot = position - logical_block * page_size
    physical_page = tl.load(block_table + batch_index * max_blocks + logical_block)

    d = tl.arange(0, block_d)
    mask = (d < head_dim) & (physical_page >= 0) & (physical_page < num_pages)
    source = ((batch_index * kv_heads + head) * tokens + token) * head_dim + d
    target = ((physical_page * kv_heads + head) * page_size + slot) * head_dim + d
    k = tl.load(new_k + source, mask=mask)
    v = tl.load(new_v + source, mask=mask)
    tl.store(page_k + target, k, mask=mask)
    tl.store(page_v + target, v, mask=mask)


@triton.jit
def _paged_kv_read_kernel(
    page_k,
    page_v,
    positions,
    block_table,
    out_k,
    out_v,
    batch: tl.constexpr,
    kv_heads: tl.constexpr,
    tokens: tl.constexpr,
    num_pages: tl.constexpr,
    max_blocks: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    token = row % tokens
    tmp = row // tokens
    head = tmp % kv_heads
    batch_index = tmp // kv_heads

    position = tl.load(positions + batch_index * tokens + token)
    logical_block = position // page_size
    slot = position - logical_block * page_size
    physical_page = tl.load(block_table + batch_index * max_blocks + logical_block)

    d = tl.arange(0, block_d)
    mask = (d < head_dim) & (physical_page >= 0) & (physical_page < num_pages)
    source = ((physical_page * kv_heads + head) * page_size + slot) * head_dim + d
    target = ((batch_index * kv_heads + head) * tokens + token) * head_dim + d
    k = tl.load(page_k + source, mask=mask)
    v = tl.load(page_v + source, mask=mask)
    tl.store(out_k + target, k, mask=mask)
    tl.store(out_v + target, v, mask=mask)


def _validate_common(
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    check_bounds: bool,
) -> tuple[int, int, int, int, int, int]:
    if not (page_k.is_cuda and page_v.is_cuda and positions.is_cuda and block_table.is_cuda):
        raise ValueError("Triton paged KV tensors must be CUDA tensors")
    if page_k.shape != page_v.shape or page_k.ndim != 4:
        raise ValueError("page K/V must be matching [pages, kv_heads, page_size, head_dim]")
    if not (page_k.is_contiguous() and page_v.is_contiguous()):
        raise ValueError("physical page tensors must be contiguous")
    if positions.ndim != 2 or positions.dtype != torch.int64 or not positions.is_contiguous():
        raise ValueError("positions must be contiguous CUDA int64 [batch, tokens]")
    if block_table.ndim != 2 or block_table.dtype != torch.int64 or not block_table.is_contiguous():
        raise ValueError("block_table must be contiguous CUDA int64 [batch, max_blocks]")

    num_pages, kv_heads, stored_page_size, head_dim = map(int, page_k.shape)
    batch, tokens = map(int, positions.shape)
    max_blocks = int(block_table.shape[1])
    if int(block_table.shape[0]) != batch:
        raise ValueError("block_table batch must match positions")
    if stored_page_size != page_size:
        raise ValueError("page_size argument must match physical page tensor")
    if head_dim not in (64, 128):
        raise ValueError("v0 paged KV supports head_dim 64 or 128")
    if check_bounds:
        validate_positions(positions, max_blocks * page_size)
        validate_block_table(block_table, num_pages)
        logical_blocks = torch.div(positions, page_size, rounding_mode="floor")
        physical = block_table.gather(1, logical_blocks)
        if bool((physical < 0).any().item()) or bool((physical >= num_pages).any().item()):
            raise IndexError("position maps to an unallocated/out-of-range physical page")
    return batch, kv_heads, tokens, num_pages, max_blocks, head_dim


def paged_kv_update(
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    check_bounds: bool = True,
) -> None:
    batch, kv_heads, tokens, num_pages, max_blocks, head_dim = _validate_common(
        page_k, page_v, positions, block_table,
        page_size=page_size, check_bounds=check_bounds,
    )
    if new_k.shape != new_v.shape or tuple(new_k.shape) != (batch, kv_heads, tokens, head_dim):
        raise ValueError("new K/V must be matching [batch, kv_heads, tokens, head_dim]")
    if new_k.dtype != page_k.dtype or new_v.dtype != page_k.dtype:
        raise TypeError("new K/V dtype must match page cache")
    if not (new_k.is_cuda and new_v.is_cuda and new_k.is_contiguous() and new_v.is_contiguous()):
        raise ValueError("new K/V must be contiguous CUDA tensors")

    block_d = triton.next_power_of_2(head_dim)
    _paged_kv_update_kernel[(batch * kv_heads * tokens,)](
        page_k, page_v, new_k, new_v, positions, block_table,
        batch=batch, kv_heads=kv_heads, tokens=tokens,
        num_pages=num_pages, max_blocks=max_blocks,
        page_size=page_size, head_dim=head_dim, block_d=block_d,
        num_warps=4,
    )


def paged_kv_read(
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    check_bounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, kv_heads, tokens, num_pages, max_blocks, head_dim = _validate_common(
        page_k, page_v, positions, block_table,
        page_size=page_size, check_bounds=check_bounds,
    )
    out_k = torch.empty(
        (batch, kv_heads, tokens, head_dim), device=page_k.device, dtype=page_k.dtype
    )
    out_v = torch.empty_like(out_k)
    block_d = triton.next_power_of_2(head_dim)
    _paged_kv_read_kernel[(batch * kv_heads * tokens,)](
        page_k, page_v, positions, block_table, out_k, out_v,
        batch=batch, kv_heads=kv_heads, tokens=tokens,
        num_pages=num_pages, max_blocks=max_blocks,
        page_size=page_size, head_dim=head_dim, block_d=block_d,
        num_warps=4,
    )
    return out_k, out_v

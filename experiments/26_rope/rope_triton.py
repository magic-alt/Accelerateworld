from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    total_units: tl.constexpr,
    sequence: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    pair_count: tl.constexpr,
    units_per_head: tl.constexpr,
    cache_stride: tl.constexpr,
    layout: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total_units
    row_head = offsets // units_per_head
    local = offsets - row_head * units_per_head
    pair_mask = active & (local < pair_count)
    tail_mask = active & (local >= pair_count)

    token = row_head // heads
    head = row_head - token * heads
    base = row_head * head_dim
    pair = local

    if layout == 0:
        dim0 = pair * 2
        dim1 = dim0 + 1
    else:
        dim0 = pair
        dim1 = pair + pair_count

    pos = tl.load(pos_ptr + token, mask=pair_mask, other=0)
    cache_offset = pos * cache_stride + pair
    c = tl.load(cos_ptr + cache_offset, mask=pair_mask, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + cache_offset, mask=pair_mask, other=0.0).to(tl.float32)

    x0 = tl.load(x_ptr + base + dim0, mask=pair_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + base + dim1, mask=pair_mask, other=0.0).to(tl.float32)
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    tl.store(out_ptr + base + dim0, y0, mask=pair_mask)
    tl.store(out_ptr + base + dim1, y1, mask=pair_mask)

    tail_dim = rotary_dim + (local - pair_count)
    tail_value = tl.load(x_ptr + base + tail_dim, mask=tail_mask, other=0.0)
    tl.store(out_ptr + base + tail_dim, tail_value, mask=tail_mask)


def rope_tensor_into(
    x: torch.Tensor,
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    layout: str,
) -> None:
    if x.ndim != 4 or out.shape != x.shape:
        raise ValueError("x/out must be matching [batch, sequence, heads, head_dim]")
    if not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("x/out must be contiguous")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("invalid rotary_dim")
    if layout not in ("interleaved", "half_split"):
        raise ValueError(f"unknown layout: {layout}")

    batch, sequence, heads, head_dim = x.shape
    pair_count = rotary_dim // 2
    units_per_head = pair_count + (head_dim - rotary_dim)
    total_units = batch * sequence * heads * units_per_head
    grid = (triton.cdiv(total_units, 256),)
    _rope_kernel[grid](
        x,
        out,
        cos,
        sin,
        position_ids,
        total_units=total_units,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        pair_count=pair_count,
        units_per_head=units_per_head,
        cache_stride=cos.shape[1],
        layout=0 if layout == "interleaved" else 1,
        BLOCK=256,
        num_warps=4,
    )


def rope_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    out = torch.empty_like(x)
    rope_tensor_into(x, out, cos, sin, position_ids, rotary_dim, layout)
    return out


def rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rope_tensor_into(q, q_out, cos, sin, position_ids, rotary_dim, layout)
    rope_tensor_into(k, k_out, cos, sin, position_ids, rotary_dim, layout)
    return q_out, k_out

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_oq,
    stride_od,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    query_length: tl.constexpr,
    key_length,
    head_dim: tl.constexpr,
    query_start,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    q_index = row % query_length
    tmp = row // query_length
    q_head = tmp % q_heads
    batch = tmp // q_heads
    group_size: tl.constexpr = q_heads // kv_heads
    kv_head = q_head // group_size

    d = tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    q_base = q_ptr + batch * stride_qb + q_head * stride_qh + q_index * stride_qq
    q_vec = tl.load(q_base + d * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

    running_m = -float("inf")
    running_l = 0.0
    output_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_K)
    absolute_query = query_start + q_index

    for key_start in tl.range(0, key_length, BLOCK_K):
        keys = key_start + key_offsets
        valid = keys < key_length
        if CAUSAL:
            valid = valid & (keys <= absolute_query)

        k_base = k_ptr + batch * stride_kb + kv_head * stride_kh
        k_ptrs = k_base + keys[:, None] * stride_kk + d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * scale
        scores = tl.where(valid, scores, -float("inf"))

        has_valid = tl.sum(valid.to(tl.int32), axis=0) > 0
        tile_m_raw = tl.max(scores, axis=0)
        tile_m = tl.where(has_valid, tile_m_raw, 0.0)
        p = tl.where(valid, tl.exp(scores - tile_m), 0.0)
        tile_l = tl.sum(p, axis=0)

        new_m = tl.where(has_valid, tl.maximum(running_m, tile_m_raw), running_m)
        alpha = tl.where(has_valid, tl.exp(running_m - new_m), 1.0)
        beta_scale = tl.where(has_valid, tl.exp(tile_m - new_m), 0.0)

        v_base = v_ptr + batch * stride_vb + kv_head * stride_vh
        v_ptrs = v_base + keys[:, None] * stride_vk + d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        weighted_v = tl.sum((p * beta_scale)[:, None] * v_tile, axis=0)

        output_acc = output_acc * alpha + weighted_v
        running_l = running_l * alpha + tile_l * beta_scale
        running_m = new_m

    out_base = out_ptr + batch * stride_ob + q_head * stride_oh + q_index * stride_oq
    tl.store(out_base + d * stride_od, output_acc / running_l, mask=d_mask)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    query_start: int,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected Q [B,QH,Q,D], K/V [B,KVH,K,D]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K/V batch and head_dim must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    if q.shape[-1] not in (64, 128):
        raise ValueError("v0 supports head_dim 64 or 128")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Q/K/V must be CUDA tensors")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")
    if causal and (query_start < 0 or query_start + q.shape[2] > k.shape[2]):
        raise ValueError("causal query positions must fit key_length")

    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(q.shape[-1])
    rows = q.shape[0] * q.shape[1] * q.shape[2]
    _flash_attention_kernel[(rows,)](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        q_heads=q.shape[1],
        kv_heads=k.shape[1],
        query_length=q.shape[2],
        key_length=k.shape[2],
        head_dim=q.shape[3],
        query_start=query_start,
        scale=1.0 / math.sqrt(q.shape[-1]),
        CAUSAL=causal,
        BLOCK_K=16,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out

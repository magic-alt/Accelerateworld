from __future__ import annotations

import math

import torch


def expand_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, q_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("K/V must have matching [B, KVH, K, D] shapes")
    kv_heads = k.shape[1]
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group = q_heads // kv_heads
    if group == 1:
        return k, v
    return k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)


def causal_mask(query_length: int, key_length: int, query_start: int, device: torch.device) -> torch.Tensor:
    if query_start < 0 or query_start + query_length > key_length:
        raise ValueError("causal query positions must fit key_length")
    q = query_start + torch.arange(query_length, device=device).view(query_length, 1)
    k = torch.arange(key_length, device=device).view(1, key_length)
    return k <= q


def manual_attention_fp32(
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
    k_exp, v_exp = expand_kv_for_gqa(k, v, q.shape[1])
    qf = q.float()
    kf = k_exp.float()
    vf = v_exp.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        mask = causal_mask(q.shape[2], k.shape[2], query_start, q.device)
        scores = scores.masked_fill(~mask.view(1, 1, q.shape[2], k.shape[2]), float("-inf"))
    row_max = scores.amax(dim=-1, keepdim=True)
    numerator = torch.exp(scores - row_max)
    probabilities = numerator / numerator.sum(dim=-1, keepdim=True)
    return torch.matmul(probabilities, vf)


def normalized_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.float()
    got = actual.float()
    return float(((got - ref).abs() / (1.0 + ref.abs())).max().item())

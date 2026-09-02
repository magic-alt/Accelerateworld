from __future__ import annotations

import torch


def causal_mask(query_length: int, key_length: int, query_start: int, device: torch.device) -> torch.Tensor:
    q = torch.arange(query_length, device=device, dtype=torch.int64) + int(query_start)
    k = torch.arange(key_length, device=device, dtype=torch.int64)
    return k.unsqueeze(0) <= q.unsqueeze(1)


def stable_softmax_fp32(
    scores: torch.Tensor,
    *,
    causal: bool,
    query_start: int,
) -> torch.Tensor:
    if scores.ndim != 4:
        raise ValueError("scores must be [batch, heads, query_length, key_length]")
    x = scores.float()
    if causal:
        mask = causal_mask(x.shape[2], x.shape[3], query_start, x.device)
        x = x.masked_fill(~mask.view(1, 1, x.shape[2], x.shape[3]), float("-inf"))
    row_max = x.amax(dim=-1, keepdim=True)
    numerator = torch.exp(x - row_max)
    denominator = numerator.sum(dim=-1, keepdim=True)
    return numerator / denominator


def low_precision_reference(
    scores: torch.Tensor,
    *,
    causal: bool,
    query_start: int,
) -> torch.Tensor:
    return stable_softmax_fp32(scores, causal=causal, query_start=query_start).to(scores.dtype)


def row_sum_error(output: torch.Tensor) -> float:
    return float((output.float().sum(dim=-1) - 1.0).abs().max().item())


def masked_max_abs(output: torch.Tensor, *, causal: bool, query_start: int) -> float:
    if not causal:
        return 0.0
    mask = causal_mask(output.shape[2], output.shape[3], query_start, output.device)
    masked = output.float().masked_select(~mask.view(1, 1, output.shape[2], output.shape[3]))
    return 0.0 if masked.numel() == 0 else float(masked.abs().max().item())

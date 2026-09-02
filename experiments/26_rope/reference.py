from __future__ import annotations

import torch


DEFAULT_THETA = 10000.0


def build_rope_cache(max_position: int, rotary_dim: int, *, device: torch.device | str, theta: float = DEFAULT_THETA) -> tuple[torch.Tensor, torch.Tensor]:
    if max_position < 0:
        raise ValueError("max_position must be non-negative")
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive and even")
    freq_index = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    inv_freq = theta ** (-freq_index / rotary_dim)
    positions = torch.arange(max_position + 1, device=device, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def build_position_ids(batch: int, sequence: int, position_start: int, *, device: torch.device | str) -> torch.Tensor:
    base = torch.arange(position_start, position_start + sequence, device=device, dtype=torch.int64)
    return base.unsqueeze(0).expand(batch, sequence).contiguous()


def _check_contract(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int) -> None:
    if x.dim() != 4:
        raise ValueError("x must have shape [batch, sequence, heads, head_dim]")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.size(-1):
        raise ValueError("rotary_dim must be positive, even and <= head_dim")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("cos/sin cache must be float32")
    if cos.dim() != 2 or sin.shape != cos.shape or cos.size(1) < rotary_dim // 2:
        raise ValueError("cos/sin cache must have shape [max_position, rotary_dim/2 or larger]")
    if position_ids.dtype != torch.int64 or position_ids.shape != x.shape[:2]:
        raise ValueError("position_ids must be int64 [batch, sequence]")


def rope_tensor_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int, layout: str) -> torch.Tensor:
    _check_contract(x, cos, sin, position_ids, rotary_dim)
    pair_count = rotary_dim // 2
    selected_cos = cos[position_ids, :pair_count].unsqueeze(2)
    selected_sin = sin[position_ids, :pair_count].unsqueeze(2)
    prefix = x[..., :rotary_dim].float()

    if layout == "interleaved":
        pairs = prefix.reshape(*prefix.shape[:-1], pair_count, 2)
        x0 = pairs[..., 0]
        x1 = pairs[..., 1]
        rotated = torch.stack(
            (x0 * selected_cos - x1 * selected_sin, x0 * selected_sin + x1 * selected_cos),
            dim=-1,
        ).flatten(-2)
    elif layout == "half_split":
        x0 = prefix[..., :pair_count]
        x1 = prefix[..., pair_count:]
        rotated = torch.cat(
            (x0 * selected_cos - x1 * selected_sin, x0 * selected_sin + x1 * selected_cos),
            dim=-1,
        )
    else:
        raise ValueError(f"unknown RoPE layout: {layout}")

    if rotary_dim == x.size(-1):
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:].float()), dim=-1)


def rope_tensor_mixed(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int, layout: str) -> torch.Tensor:
    return rope_tensor_fp32(x, cos, sin, position_ids, rotary_dim, layout).to(x.dtype)


def rope_qk_fp32(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[:2] != k.shape[:2] or q.size(-1) != k.size(-1):
        raise ValueError("q and k must share batch/sequence/head_dim")
    return (
        rope_tensor_fp32(q, cos, sin, position_ids, rotary_dim, layout),
        rope_tensor_fp32(k, cos, sin, position_ids, rotary_dim, layout),
    )


def rope_qk_mixed(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    q_ref, k_ref = rope_qk_fp32(q, k, cos, sin, position_ids, rotary_dim, layout)
    return q_ref.to(q.dtype), k_ref.to(k.dtype)

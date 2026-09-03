from __future__ import annotations

import torch
import torch.nn.functional as F

from lab_config import format_info, parameter_count, validate_granularity


def round_half_away_from_zero_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _group_view(x: torch.Tensor, group_size: int, fill: float) -> torch.Tensor:
    rows, cols = map(int, x.shape)
    groups = (cols + group_size - 1) // group_size
    padded_cols = groups * group_size
    if padded_cols != cols:
        x = F.pad(x, (0, padded_cols - cols), value=fill)
    return x.view(rows, groups, group_size)


def compute_qparams(x: torch.Tensor, *, format_name: str, granularity: str, group_size: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("x must be a rank-2 floating tensor")
    validate_granularity(granularity, group_size)
    info = format_info(format_name)
    xf = x.float()
    if granularity == "per_tensor":
        minimum = xf.min().view(1)
        maximum = xf.max().view(1)
        max_abs = xf.abs().max().view(1)
    elif granularity == "per_channel":
        minimum = xf.amin(dim=1)
        maximum = xf.amax(dim=1)
        max_abs = xf.abs().amax(dim=1)
    else:
        groups = (int(x.shape[1]) + group_size - 1) // group_size
        max_abs = _group_view(xf.abs(), group_size, 0.0).amax(dim=2).reshape(-1)
        minimum = _group_view(xf, group_size, float("inf")).amin(dim=2).reshape(-1)
        maximum = _group_view(xf, group_size, float("-inf")).amax(dim=2).reshape(-1)
        assert max_abs.numel() == int(x.shape[0]) * groups
    if info["asymmetric"]:
        qmin, qmax = int(info["qmin"]), int(info["qmax"])
        scale = (maximum - minimum) / float(qmax - qmin)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zp = round_half_away_from_zero_tensor(qmin - minimum / scale)
        zp = zp.clamp(qmin, qmax).to(torch.int32)
    else:
        scale = max_abs / float(info["qmax"])
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zp = torch.zeros_like(scale, dtype=torch.int32)
    expected = parameter_count(int(x.shape[0]), int(x.shape[1]), granularity, group_size)
    if scale.numel() != expected:
        raise AssertionError(f"qparam count mismatch: {scale.numel()} != {expected}")
    return scale.contiguous().float(), zp.contiguous()


def _expanded_qparams(scales: torch.Tensor, zero_points: torch.Tensor, rows: int, cols: int, granularity: str, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if granularity == "per_tensor":
        return scales.view(1, 1), zero_points.view(1, 1)
    if granularity == "per_channel":
        return scales.view(rows, 1), zero_points.view(rows, 1)
    groups = (cols + group_size - 1) // group_size
    s = scales.view(rows, groups).repeat_interleave(group_size, dim=1)[:, :cols]
    z = zero_points.view(rows, groups).repeat_interleave(group_size, dim=1)[:, :cols]
    return s, z


def quantize_reference(x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, *, format_name: str, granularity: str, group_size: int = 0) -> torch.Tensor:
    rows, cols = map(int, x.shape)
    info = format_info(format_name)
    s, z = _expanded_qparams(scales, zero_points, rows, cols, granularity, group_size)
    q = round_half_away_from_zero_tensor(x.float() / s + z.float())
    q = q.clamp(int(info["qmin"]), int(info["qmax"])).to(torch.int8)
    if not info["packed"]:
        return q.contiguous()
    low = (q[:, 0::2].to(torch.int16) & 0xF).to(torch.uint8)
    high_q = q[:, 1::2]
    if high_q.shape[1] < low.shape[1]:
        high_q = F.pad(high_q, (0, 1), value=0)
    high = ((high_q.to(torch.int16) & 0xF) << 4).to(torch.uint8)
    return (low | high).contiguous()


def unpack_int4_tensor(packed: torch.Tensor, cols: int) -> torch.Tensor:
    low = (packed & 0xF).to(torch.int16)
    high = ((packed >> 4) & 0xF).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int16, device=packed.device)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :cols].to(torch.int8).contiguous()


def dequantize_reference(q: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, *, rows: int, cols: int, format_name: str, granularity: str, group_size: int = 0, output_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    info = format_info(format_name)
    qi = unpack_int4_tensor(q, cols) if info["packed"] else q.to(torch.int8)
    if tuple(qi.shape) != (rows, cols):
        raise ValueError("quantized tensor shape does not match rows/cols")
    s, z = _expanded_qparams(scales, zero_points, rows, cols, granularity, group_size)
    return ((qi.float() - z.float()) * s).to(output_dtype).contiguous()

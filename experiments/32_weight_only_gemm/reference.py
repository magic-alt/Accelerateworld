from __future__ import annotations

import torch
import torch.nn.functional as F

from lab_config import qparam_count, validate_format, validate_granularity

QBOUNDS = {
    "int8_sym": (-127, 127, False),
    "int8_asym": (-128, 127, True),
    "int4_sym": (-8, 7, False),
}

def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

def _group_view(x: torch.Tensor, group_size: int, fill: float) -> torch.Tensor:
    n, k = map(int, x.shape)
    groups = (k + group_size - 1) // group_size
    padded = groups * group_size
    if padded != k:
        x = F.pad(x, (0, padded-k), value=fill)
    return x.view(n, groups, group_size)

def compute_weight_qparams(weight: torch.Tensor, *, format_name: str, granularity: str, group_size: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    validate_format(format_name)
    validate_granularity(granularity, group_size)
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be floating [N,K]")
    w = weight.float()
    qmin, qmax, asymmetric = QBOUNDS[format_name]
    if granularity == "per_tensor":
        minimum, maximum, max_abs = w.min().view(1), w.max().view(1), w.abs().max().view(1)
    elif granularity == "per_channel":
        minimum, maximum, max_abs = w.amin(1), w.amax(1), w.abs().amax(1)
    else:
        minimum = _group_view(w, group_size, float("inf")).amin(2).reshape(-1)
        maximum = _group_view(w, group_size, float("-inf")).amax(2).reshape(-1)
        max_abs = _group_view(w.abs(), group_size, 0.0).amax(2).reshape(-1)
    if asymmetric:
        scales = (maximum-minimum) / float(qmax-qmin)
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        zero = round_half_away_from_zero(qmin - minimum/scales).clamp(qmin,qmax).to(torch.int32)
    else:
        scales = max_abs / float(qmax)
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        zero = torch.zeros_like(scales, dtype=torch.int32)
    expected=qparam_count(int(weight.shape[0]), int(weight.shape[1]), granularity, group_size)
    if scales.numel()!=expected:
        raise AssertionError("qparam count mismatch")
    return scales.float().contiguous(), zero.contiguous()

def _expanded(scales: torch.Tensor, zero: torch.Tensor, n: int, k: int, granularity: str, group_size: int):
    if granularity=="per_tensor":
        return scales.view(1,1), zero.view(1,1)
    if granularity=="per_channel":
        return scales.view(n,1), zero.view(n,1)
    groups=(k+group_size-1)//group_size
    s=scales.view(n,groups).repeat_interleave(group_size,1)[:,:k]
    z=zero.view(n,groups).repeat_interleave(group_size,1)[:,:k]
    return s,z

def quantize_weight_reference(weight: torch.Tensor, scales: torch.Tensor, zero: torch.Tensor, *, format_name: str, granularity: str, group_size: int = 0) -> torch.Tensor:
    n,k=map(int,weight.shape)
    qmin,qmax,_=QBOUNDS[format_name]
    s,z=_expanded(scales,zero,n,k,granularity,group_size)
    q=round_half_away_from_zero(weight.float()/s+z.float()).clamp(qmin,qmax).to(torch.int8)
    if format_name!="int4_sym":
        return q.contiguous()
    low=(q[:,0::2].to(torch.int16)&0xF).to(torch.uint8)
    high_q=q[:,1::2]
    if high_q.shape[1]<low.shape[1]:
        high_q=F.pad(high_q,(0,1),value=0)
    high=((high_q.to(torch.int16)&0xF)<<4).to(torch.uint8)
    return (low|high).contiguous()

def unpack_int4(packed: torch.Tensor, k: int) -> torch.Tensor:
    low=(packed&0xF).to(torch.int16)
    high=((packed>>4)&0xF).to(torch.int16)
    low=torch.where(low>=8,low-16,low)
    high=torch.where(high>=8,high-16,high)
    out=torch.empty((packed.shape[0],packed.shape[1]*2),dtype=torch.int16,device=packed.device)
    out[:,0::2]=low
    out[:,1::2]=high
    return out[:,:k].to(torch.int8).contiguous()

def dequantize_weight_reference(qweight: torch.Tensor, scales: torch.Tensor, zero: torch.Tensor, *, n: int, k: int, format_name: str, granularity: str, group_size: int = 0) -> torch.Tensor:
    qi=unpack_int4(qweight,k) if format_name=="int4_sym" else qweight.to(torch.int8)
    if tuple(qi.shape)!=(n,k):
        raise ValueError("qweight shape mismatch")
    s,z=_expanded(scales,zero,n,k,granularity,group_size)
    return ((qi.float()-z.float())*s).contiguous()

def gemm_reference(activation: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zero: torch.Tensor, *, format_name: str, granularity: str, group_size: int = 0, output_dtype: torch.dtype | None = None) -> torch.Tensor:
    if activation.ndim!=2 or not activation.is_floating_point():
        raise ValueError("activation must be floating [M,K]")
    _,k=map(int,activation.shape)
    n=int(qweight.shape[0])
    weight=dequantize_weight_reference(qweight,scales,zero,n=n,k=k,format_name=format_name,granularity=granularity,group_size=group_size)
    out=activation.float().matmul(weight.t())
    return out.to(output_dtype or activation.dtype).contiguous()

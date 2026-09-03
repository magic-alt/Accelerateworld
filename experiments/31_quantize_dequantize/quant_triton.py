from __future__ import annotations

import torch
import triton
import triton.language as tl
from lab_config import granularity_id, parameter_count

@triton.jit
def _quantize_int8_kernel(x_ptr, scales_ptr, zero_ptr, out_ptr, rows: tl.constexpr, cols: tl.constexpr, granularity: tl.constexpr, group_size: tl.constexpr, groups_per_row: tl.constexpr, qmin: tl.constexpr, qmax: tl.constexpr, block: tl.constexpr):
    offsets=tl.program_id(0)*block+tl.arange(0,block); total=rows*cols; mask=offsets<total; row=offsets//cols; col=offsets-row*cols
    if granularity==0: pidx=tl.zeros_like(offsets)
    elif granularity==1: pidx=row
    else: pidx=row*groups_per_row+col//group_size
    x=tl.load(x_ptr+offsets,mask=mask,other=0.0).to(tl.float32); scale=tl.load(scales_ptr+pidx,mask=mask,other=1.0); zp=tl.load(zero_ptr+pidx,mask=mask,other=0).to(tl.float32)
    qf=x/scale+zp; rounded=tl.where(qf>=0.0,tl.floor(qf+0.5),tl.ceil(qf-0.5)); q=tl.maximum(qmin,tl.minimum(qmax,rounded)); tl.store(out_ptr+offsets,q,mask=mask)

@triton.jit
def _dequantize_int8_kernel(q_ptr,scales_ptr,zero_ptr,out_ptr,rows:tl.constexpr,cols:tl.constexpr,granularity:tl.constexpr,group_size:tl.constexpr,groups_per_row:tl.constexpr,block:tl.constexpr):
    offsets=tl.program_id(0)*block+tl.arange(0,block); mask=offsets<rows*cols; row=offsets//cols; col=offsets-row*cols
    if granularity==0: pidx=tl.zeros_like(offsets)
    elif granularity==1: pidx=row
    else: pidx=row*groups_per_row+col//group_size
    q=tl.load(q_ptr+offsets,mask=mask,other=0).to(tl.float32); s=tl.load(scales_ptr+pidx,mask=mask,other=1.0); z=tl.load(zero_ptr+pidx,mask=mask,other=0).to(tl.float32); tl.store(out_ptr+offsets,(q-z)*s,mask=mask)

@triton.jit
def _quantize_int4_kernel(x_ptr,scales_ptr,out_ptr,rows:tl.constexpr,cols:tl.constexpr,granularity:tl.constexpr,group_size:tl.constexpr,groups_per_row:tl.constexpr,packed_cols:tl.constexpr,block:tl.constexpr):
    b=tl.program_id(0)*block+tl.arange(0,block); mask=b<rows*packed_cols; row=b//packed_cols; bc=b-row*packed_cols; c0=bc*2; c1=c0+1; e0=row*cols+c0; e1=row*cols+c1
    if granularity==0: p0=tl.zeros_like(b); p1=p0
    elif granularity==1: p0=row; p1=row
    else: p0=row*groups_per_row+c0//group_size; p1=row*groups_per_row+c1//group_size
    v0=mask&(c0<cols); v1=mask&(c1<cols); x0=tl.load(x_ptr+e0,mask=v0,other=0.0).to(tl.float32); x1=tl.load(x_ptr+e1,mask=v1,other=0.0).to(tl.float32); s0=tl.load(scales_ptr+p0,mask=v0,other=1.0); s1=tl.load(scales_ptr+p1,mask=v1,other=1.0)
    a=x0/s0; d=x1/s1; r0=tl.where(a>=0.0,tl.floor(a+0.5),tl.ceil(a-0.5)); r1=tl.where(d>=0.0,tl.floor(d+0.5),tl.ceil(d-0.5)); q0=tl.maximum(-8,tl.minimum(7,r0)).to(tl.int32); q1=tl.maximum(-8,tl.minimum(7,r1)).to(tl.int32); n0=q0&15; n1=tl.where(v1,q1&15,0); tl.store(out_ptr+b,n0|(n1<<4),mask=mask)

@triton.jit
def _dequantize_int4_kernel(packed_ptr,scales_ptr,out_ptr,rows:tl.constexpr,cols:tl.constexpr,granularity:tl.constexpr,group_size:tl.constexpr,groups_per_row:tl.constexpr,packed_cols:tl.constexpr,block:tl.constexpr):
    o=tl.program_id(0)*block+tl.arange(0,block); mask=o<rows*cols; row=o//cols; col=o-row*cols; byte=tl.load(packed_ptr+row*packed_cols+col//2,mask=mask,other=0).to(tl.int32); nibble=tl.where((col&1)==0,byte&15,(byte>>4)&15); q=tl.where(nibble>=8,nibble-16,nibble).to(tl.float32)
    if granularity==0: pidx=tl.zeros_like(o)
    elif granularity==1: pidx=row
    else: pidx=row*groups_per_row+col//group_size
    s=tl.load(scales_ptr+pidx,mask=mask,other=1.0); tl.store(out_ptr+o,q*s,mask=mask)

def _validate(scales,zero,rows,cols,granularity,group_size):
    if not(scales.is_cuda and zero.is_cuda) or scales.dtype!=torch.float32 or zero.dtype!=torch.int32: raise ValueError("qparams must be CUDA float32/int32")
    expected=parameter_count(rows,cols,granularity,group_size)
    if scales.numel()!=expected or zero.numel()!=expected: raise ValueError("qparam count mismatch")
    return granularity_id(granularity),(cols+max(group_size,1)-1)//max(group_size,1)

def quantize_int8(x,scales,zero_points,*,granularity,group_size,asymmetric):
    rows,cols=map(int,x.shape); gid,groups=_validate(scales,zero_points,rows,cols,granularity,group_size); out=torch.empty((rows,cols),device=x.device,dtype=torch.int8); block=256
    _quantize_int8_kernel[(triton.cdiv(rows*cols,block),)](x,scales,zero_points,out,rows=rows,cols=cols,granularity=gid,group_size=max(group_size,1),groups_per_row=groups,qmin=-128 if asymmetric else -127,qmax=127,block=block); return out

def dequantize_int8(q,scales,zero_points,*,granularity,group_size,output_dtype):
    rows,cols=map(int,q.shape); gid,groups=_validate(scales,zero_points,rows,cols,granularity,group_size); out=torch.empty((rows,cols),device=q.device,dtype=output_dtype); block=256
    _dequantize_int8_kernel[(triton.cdiv(rows*cols,block),)](q,scales,zero_points,out,rows=rows,cols=cols,granularity=gid,group_size=max(group_size,1),groups_per_row=groups,block=block); return out

def quantize_int4(x,scales,*,granularity,group_size):
    rows,cols=map(int,x.shape); zero=torch.zeros_like(scales,dtype=torch.int32); gid,groups=_validate(scales,zero,rows,cols,granularity,group_size); pc=(cols+1)//2; out=torch.empty((rows,pc),device=x.device,dtype=torch.uint8); block=256
    _quantize_int4_kernel[(triton.cdiv(rows*pc,block),)](x,scales,out,rows=rows,cols=cols,granularity=gid,group_size=max(group_size,1),groups_per_row=groups,packed_cols=pc,block=block); return out

def dequantize_int4(q,scales,*,rows,cols,granularity,group_size,output_dtype):
    zero=torch.zeros_like(scales,dtype=torch.int32); gid,groups=_validate(scales,zero,rows,cols,granularity,group_size); out=torch.empty((rows,cols),device=q.device,dtype=output_dtype); block=256
    _dequantize_int4_kernel[(triton.cdiv(rows*cols,block),)](q,scales,out,rows=rows,cols=cols,granularity=gid,group_size=max(group_size,1),groups_per_row=groups,packed_cols=(cols+1)//2,block=block); return out

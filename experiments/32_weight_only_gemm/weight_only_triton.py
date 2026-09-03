from __future__ import annotations

import torch
import triton
import triton.language as tl

from lab_config import qparam_count, validate_format, validate_granularity

@triton.jit
def _weight_only_gemm_kernel(
    a_ptr, q_ptr, scales_ptr, zero_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    PACKED_K: tl.constexpr,
    GRANULARITY: tl.constexpr, GROUP_SIZE: tl.constexpr, GROUPS_PER_ROW: tl.constexpr,
    FORMAT_ID: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    offs_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    offs_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)

    for k0 in range(0,K,BLOCK_K):
        offs_k=k0+tl.arange(0,BLOCK_K)
        a_mask=(offs_m[:,None]<M)&(offs_k[None,:]<K)
        a=tl.load(a_ptr+offs_m[:,None]*K+offs_k[None,:],mask=a_mask,other=0.0).to(tl.float32)

        wk=offs_k[:,None]
        wn=offs_n[None,:]
        wmask=(wk<K)&(wn<N)
        if FORMAT_ID==0:
            q=tl.load(q_ptr+wn*K+wk,mask=wmask,other=0).to(tl.float32)
        else:
            packed=tl.load(q_ptr+wn*PACKED_K+(wk//2),mask=wmask,other=0).to(tl.int32)
            nibble=tl.where((wk&1)==0,packed&15,(packed>>4)&15)
            q=tl.where(nibble>=8,nibble-16,nibble).to(tl.float32)

        if GRANULARITY==0:
            pidx=tl.zeros_like(wk+wn)
        elif GRANULARITY==1:
            pidx=wn+tl.zeros_like(wk)
        else:
            pidx=wn*GROUPS_PER_ROW+(wk//GROUP_SIZE)
        scale=tl.load(scales_ptr+pidx,mask=wmask,other=1.0)
        zero=tl.load(zero_ptr+pidx,mask=wmask,other=0).to(tl.float32)
        w=(q-zero)*scale
        acc+=tl.dot(a,w,input_precision="ieee")

    out_mask=(offs_m[:,None]<M)&(offs_n[None,:]<N)
    tl.store(c_ptr+offs_m[:,None]*N+offs_n[None,:],acc,mask=out_mask)

def _validate(activation,qweight,scales,zero_points,format_name,granularity,group_size):
    validate_format(format_name); validate_granularity(granularity,group_size)
    if activation.ndim!=2 or not activation.is_cuda or activation.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError("activation must be CUDA fp16/bf16 [M,K]")
    if activation.dtype==torch.bfloat16 and torch.cuda.get_device_capability(activation.device)[0]<8:
        raise RuntimeError("BF16 weight-only GEMM requires compute capability >= 8.0")
    m,k=map(int,activation.shape); n=int(qweight.shape[0])
    if format_name=="int4_sym":
        if qweight.dtype!=torch.uint8 or tuple(qweight.shape)!=(n,(k+1)//2):
            raise ValueError("INT4 qweight must be uint8 [N,ceil(K/2)]")
    else:
        if qweight.dtype!=torch.int8 or tuple(qweight.shape)!=(n,k):
            raise ValueError("INT8 qweight must be int8 [N,K]")
    expected=qparam_count(n,k,granularity,group_size)
    if scales.dtype!=torch.float32 or zero_points.dtype!=torch.int32 or scales.numel()!=expected or zero_points.numel()!=expected:
        raise ValueError("qparam contract mismatch")
    if not all(t.is_cuda and t.is_contiguous() for t in (activation,qweight,scales,zero_points)):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    gid={"per_tensor":0,"per_channel":1,"group":2}[granularity]
    return m,n,k,gid,(k+max(group_size,1)-1)//max(group_size,1)

def weight_only_gemm(activation,qweight,scales,zero_points,*,format_name,granularity,group_size=0):
    m,n,k,gid,groups=_validate(activation,qweight,scales,zero_points,format_name,granularity,group_size)
    out=torch.empty((m,n),device=activation.device,dtype=activation.dtype)
    bm,bn,bk=16,32,32
    _weight_only_gemm_kernel[(triton.cdiv(m,bm),triton.cdiv(n,bn))](
        activation,qweight,scales,zero_points,out,
        M=m,N=n,K=k,PACKED_K=(k+1)//2,
        GRANULARITY=gid,GROUP_SIZE=max(group_size,1),GROUPS_PER_ROW=groups,
        FORMAT_ID=1 if format_name=="int4_sym" else 0,
        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,
        num_warps=4,
    )
    return out

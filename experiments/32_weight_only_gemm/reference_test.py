from __future__ import annotations

import torch

from reference import (
    compute_weight_qparams,
    dequantize_weight_reference,
    gemm_reference,
    quantize_weight_reference,
)

def run_case(format_name: str, granularity: str, group_size: int, *, n: int, k: int) -> None:
    torch.manual_seed(2026+n+k)
    weight=torch.randn(n,k,dtype=torch.float32)*0.35
    activation=torch.randn(4,k,dtype=torch.float32)
    scales,zero=compute_weight_qparams(weight,format_name=format_name,granularity=granularity,group_size=group_size)
    qweight=quantize_weight_reference(weight,scales,zero,format_name=format_name,granularity=granularity,group_size=group_size)
    deq=dequantize_weight_reference(qweight,scales,zero,n=n,k=k,format_name=format_name,granularity=granularity,group_size=group_size)
    got=gemm_reference(activation,qweight,scales,zero,format_name=format_name,granularity=granularity,group_size=group_size,output_dtype=torch.float32)
    expected=activation @ deq.t()
    torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-6)
    assert torch.isfinite(got).all()
    if format_name=="int4_sym":
        assert qweight.dtype==torch.uint8
        assert qweight.shape==(n,(k+1)//2)
    else:
        assert qweight.dtype==torch.int8
        assert qweight.shape==(n,k)

def main() -> None:
    run_case("int8_sym","per_channel",0,n=7,k=65)
    run_case("int8_asym","group",32,n=8,k=128)
    run_case("int4_sym","group",32,n=7,k=65)
    print("weight-only GEMM CPU reference validation: PASS")

if __name__=="__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))

import accelerateworld_weight_only_gemm_cuda as cuda_ext
from lab_config import (
    arithmetic_intensity,
    expected_regime,
    qparam_count,
    quantized_weight_bytes,
    shapes_for_family,
)
from reference import (
    compute_weight_qparams,
    dequantize_weight_reference,
    gemm_reference,
    quantize_weight_reference,
)
from weight_only_triton import weight_only_gemm as triton_weight_only_gemm

FORMAT_IDS={"int8_sym":0,"int8_asym":1,"int4_sym":2}
GRANULARITY_IDS={"per_tensor":0,"per_channel":1,"group":2}

def bench(fn,warmup:int,iterations:int)->float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end)/iterations

def error_metrics(got:torch.Tensor,ref:torch.Tensor)->dict:
    gf=got.float(); rf=ref.float(); diff=(gf-rf).abs()
    denom=rf.abs().amax().clamp_min(1e-6)
    return {
        "max_abs_error":float(diff.amax().item()),
        "rmse":float(torch.sqrt(torch.mean((gf-rf)**2)).item()),
        "normalized_max_error":float((diff.amax()/denom).item()),
    }

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--family",default="validation",choices=("validation","decode","prefill","all"))
    ap.add_argument("--dtypes",default="fp16,bf16")
    ap.add_argument("--formats",default="int8_sym,int4_sym")
    ap.add_argument("--granularity",default="group",choices=("per_tensor","per_channel","group"))
    ap.add_argument("--group-size",type=int,default=64)
    ap.add_argument("--warmup",type=int,default=5)
    ap.add_argument("--iterations",type=int,default=20)
    ap.add_argument("--json",dest="json_path",default="results/weight-only-gemm-validation.json")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("physical CUDA GPU required")

    records=[]
    group_size=args.group_size if args.granularity=="group" else 0
    capability=torch.cuda.get_device_capability()
    for shape in shapes_for_family(args.family):
        for dtype_name in [x.strip() for x in args.dtypes.split(",") if x.strip()]:
            if dtype_name=="bf16" and capability[0]<8:
                records.append({"shape":shape.name,"dtype":"bf16","status":"skipped","reason":"bf16 requires cc>=8.0"})
                continue
            dtype={"fp16":torch.float16,"bf16":torch.bfloat16}[dtype_name]
            torch.manual_seed(2026+shape.m+shape.n+shape.k)
            activation=(torch.randn(shape.m,shape.k,device="cuda",dtype=torch.float32)*0.5).to(dtype)
            weight=(torch.randn(shape.n,shape.k,device="cuda",dtype=torch.float32)*0.2).to(dtype)
            for fmt in [x.strip() for x in args.formats.split(",") if x.strip()]:
                scales,zero=compute_weight_qparams(weight,format_name=fmt,granularity=args.granularity,group_size=group_size)
                qweight=quantize_weight_reference(weight,scales,zero,format_name=fmt,granularity=args.granularity,group_size=group_size)
                reference=gemm_reference(
                    activation,qweight,scales,zero,
                    format_name=fmt,granularity=args.granularity,group_size=group_size,
                    output_dtype=torch.float32,
                )
                deq_weight=dequantize_weight_reference(
                    qweight,scales,zero,n=shape.n,k=shape.k,
                    format_name=fmt,granularity=args.granularity,group_size=group_size,
                ).to(dtype)

                providers={
                    "pytorch_materialized_dense":lambda: torch.mm(activation,deq_weight.t()),
                    "pytorch_dequant_then_gemm":lambda: torch.mm(
                        activation,
                        dequantize_weight_reference(
                            qweight,scales,zero,n=shape.n,k=shape.k,
                            format_name=fmt,granularity=args.granularity,group_size=group_size,
                        ).to(dtype).t(),
                    ),
                    "triton_weight_only":lambda: triton_weight_only_gemm(
                        activation,qweight,scales,zero,
                        format_name=fmt,granularity=args.granularity,group_size=group_size,
                    ),
                    "cuda_weight_only":lambda: cuda_ext.weight_only_gemm(
                        activation,qweight,scales,zero,
                        FORMAT_IDS[fmt],GRANULARITY_IDS[args.granularity],group_size,
                    ),
                }
                pcount=qparam_count(shape.n,shape.k,args.granularity,group_size)
                logical_bytes=(
                    activation.numel()*activation.element_size()
                    + quantized_weight_bytes(fmt,shape.n,shape.k)
                    + pcount*(4+4)
                    + shape.m*shape.n*activation.element_size()
                )
                provider_rows=[]
                for name,fn in providers.items():
                    got=fn(); torch.cuda.synchronize()
                    metrics=error_metrics(got,reference)
                    latency=bench(fn,args.warmup,args.iterations)
                    provider_rows.append({
                        "provider":name,
                        "latency_ms":latency,
                        "logical_gbps":logical_bytes/(latency*1e6),
                        "logical_tflops":shape.flops/(latency*1e9),
                        **metrics,
                    })
                winner=min(provider_rows,key=lambda r:r["latency_ms"])["provider"]
                records.append({
                    "shape":shape.name,"role":shape.role,
                    "m":shape.m,"n":shape.n,"k":shape.k,
                    "dtype":dtype_name,"format":fmt,
                    "granularity":args.granularity,"group_size":group_size,
                    "qparam_count":pcount,
                    "quantized_weight_bytes":quantized_weight_bytes(fmt,shape.n,shape.k),
                    "fp16_weight_bytes":shape.n*shape.k*2,
                    "weight_compression_ratio":(shape.n*shape.k*2)/quantized_weight_bytes(fmt,shape.n,shape.k),
                    "logical_arithmetic_intensity_flops_per_byte":arithmetic_intensity(shape,fmt),
                    "expected_regime":expected_regime(shape),
                    "winner":winner,
                    "providers":provider_rows,
                })

    payload={
        "schema":"accelerateworld.weight_only_gemm.v1",
        "generated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "gpu":torch.cuda.get_device_name(),
        "compute_capability":f"{capability[0]}.{capability[1]}",
        "torch":torch.__version__,
        "records":records,
    }
    path=Path(args.json_path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()

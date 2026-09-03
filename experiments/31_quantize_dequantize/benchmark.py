from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Callable
import torch,triton
import accelerateworld_quant_cuda as cuda_quant
from lab_config import FORMATS,GRANULARITIES,SHAPE_FAMILIES,format_info,granularity_id,shapes_for_family,storage_bytes
from quant_triton import dequantize_int4 as triton_dequant4,dequantize_int8 as triton_dequant8,quantize_int4 as triton_quant4,quantize_int8 as triton_quant8
from reference import compute_qparams,dequantize_reference,quantize_reference,unpack_int4_tensor
DTYPE_MAP={"fp16":torch.float16,"bf16":torch.bfloat16}; OUTPUT_DTYPE_ID={torch.float16:0,torch.bfloat16:1}
def bench_cuda(fn:Callable[[],object],warmup:int,iterations:int)->float:
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(iterations): fn()
    b.record(); b.synchronize(); return float(a.elapsed_time(b))/iterations
def bf16_supported():
    if not torch.cuda.is_available(): return False
    major,_=torch.cuda.get_device_capability(); return major>=8 and bool(torch.cuda.is_bf16_supported())
def provider_ops(provider,fmt,granularity,group_size,dtype):
    info=format_info(fmt); gid=granularity_id(granularity)
    if provider=="pytorch": return (lambda x,s,z:quantize_reference(x,s,z,format_name=fmt,granularity=granularity,group_size=group_size),lambda q,s,z,r,c:dequantize_reference(q,s,z,rows=r,cols=c,format_name=fmt,granularity=granularity,group_size=group_size,output_dtype=dtype))
    if provider=="triton":
        if info["packed"]: return (lambda x,s,z:triton_quant4(x,s,granularity=granularity,group_size=group_size),lambda q,s,z,r,c:triton_dequant4(q,s,rows=r,cols=c,granularity=granularity,group_size=group_size,output_dtype=dtype))
        return (lambda x,s,z:triton_quant8(x,s,z,granularity=granularity,group_size=group_size,asymmetric=bool(info["asymmetric"])),lambda q,s,z,r,c:triton_dequant8(q,s,z,granularity=granularity,group_size=group_size,output_dtype=dtype))
    if info["packed"]: return (lambda x,s,z:cuda_quant.quantize_int4(x,s,gid,group_size),lambda q,s,z,r,c:cuda_quant.dequantize_int4(q,s,r,c,gid,group_size,OUTPUT_DTYPE_ID[dtype]))
    return (lambda x,s,z:cuda_quant.quantize_int8(x,s,z,gid,group_size,bool(info["asymmetric"])),lambda q,s,z,r,c:cuda_quant.dequantize_int8(q,s,z,gid,group_size,OUTPUT_DTYPE_ID[dtype]))
def run_case(shape,dtype_name,fmt,granularity,group_size,warmup,iterations):
    dtype=DTYPE_MAP[dtype_name]; torch.manual_seed(2026); x=(torch.randn((shape.rows,shape.cols),device="cuda",dtype=dtype)*1.75).contiguous(); s,z=compute_qparams(x,format_name=fmt,granularity=granularity,group_size=group_size); ref=quantize_reference(x,s,z,format_name=fmt,granularity=granularity,group_size=group_size); records={}
    for provider in ("pytorch","triton","cuda"):
        quant,dequant=provider_ops(provider,fmt,granularity,group_size,dtype); q=quant(x,s,z); dq=dequant(q,s,z,shape.rows,shape.cols); torch.cuda.synchronize()
        if not torch.equal(q,ref): raise AssertionError(f"{provider}/{fmt}/{granularity}: storage mismatch")
        err=dq.float()-x.float(); qi=unpack_int4_tensor(q,shape.cols) if format_info(fmt)["packed"] else q; info=format_info(fmt); sat=((qi==int(info["qmin"]))|(qi==int(info["qmax"]))).float().mean(); qms=bench_cuda(lambda:quant(x,s,z),warmup,iterations); dms=bench_cuda(lambda:dequant(q,s,z,shape.rows,shape.cols),warmup,iterations); src=x.numel()*x.element_size(); qb=storage_bytes(fmt,shape.rows,shape.cols)
        records[provider]={"quantize_ms":qms,"dequantize_ms":dms,"quantize_logical_gbps":(src+qb)/(qms/1000)/1e9,"dequantize_logical_gbps":(src+qb)/(dms/1000)/1e9,"rmse":float(torch.sqrt(torch.mean(err*err)).item()),"max_abs_error":float(err.abs().amax().item()),"normalized_max_error":float((err.abs().amax()/x.float().abs().amax().clamp_min(1e-8)).item()),"saturation_rate":float(sat.item())}
    src=x.numel()*x.element_size(); qb=storage_bytes(fmt,shape.rows,shape.cols); return {"shape":{"name":shape.name,"role":shape.role,"rows":shape.rows,"cols":shape.cols},"dtype":dtype_name,"format":fmt,"granularity":granularity,"group_size":group_size,"source_bytes":src,"quantized_bytes":qb,"compression_ratio":src/qb,"qparam_count":int(s.numel()),"providers":records,"winners":{"quantize_ms":min(records,key=lambda n:records[n]["quantize_ms"]),"dequantize_ms":min(records,key=lambda n:records[n]["dequantize_ms"])}}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--family",choices=SHAPE_FAMILIES,default="validation"); p.add_argument("--dtypes",default="fp16,bf16"); p.add_argument("--formats",default=",".join(FORMATS)); p.add_argument("--granularities",default=",".join(GRANULARITIES)); p.add_argument("--group-size",type=int,default=32); p.add_argument("--warmup",type=int,default=5); p.add_argument("--iterations",type=int,default=20); p.add_argument("--json",type=Path); a=p.parse_args();
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    payload={"gpu":torch.cuda.get_device_name(),"compute_capability":".".join(map(str,torch.cuda.get_device_capability())),"torch":torch.__version__,"torch_cuda":torch.version.cuda,"triton":triton.__version__,"rounding":"round-half-away-from-zero then saturating clamp","int4_storage":"signed two's-complement nibbles; even col low nibble; odd col high nibble","results":[],"skipped":[]}
    for dtype_name in tuple(x.strip() for x in a.dtypes.split(",") if x.strip()):
        if dtype_name=="bf16" and not bf16_supported(): payload["skipped"].append({"dtype":"bf16","reason":"requires CC>=8"}); continue
        for shape in shapes_for_family(a.family):
            for fmt in tuple(x.strip() for x in a.formats.split(",") if x.strip()):
                for gran in tuple(x.strip() for x in a.granularities.split(",") if x.strip()): payload["results"].append(run_case(shape,dtype_name,fmt,gran,a.group_size if gran=="group" else 0,a.warmup,a.iterations))
    if a.json: a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print("quantize/dequantize physical-GPU validation: PASS"); return 0
if __name__=="__main__": raise SystemExit(main())

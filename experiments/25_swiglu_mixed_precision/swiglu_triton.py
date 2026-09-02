from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(
    packed_ptr,
    output_ptr,
    rows,
    intermediate,
    packed_stride_row,
    output_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = rows * intermediate
    mask = offsets < total

    row = offsets // intermediate
    col = offsets - row * intermediate
    gate_offsets = row * packed_stride_row + col
    up_offsets = gate_offsets + intermediate

    gate = tl.load(packed_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(packed_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    output = gate * sigmoid * up
    output_offsets = row * output_stride_row + col
    tl.store(output_ptr + output_offsets, output, mask=mask)


def _validate(packed: torch.Tensor, output: torch.Tensor) -> tuple[int, int]:
    if not (packed.is_cuda and output.is_cuda):
        raise ValueError("Triton SwiGLU requires CUDA tensors")
    if packed.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("packed gate/up tensor must be float16 or bfloat16")
    if output.dtype != packed.dtype:
        raise ValueError("output dtype must match packed input dtype")
    if packed.ndim != 2 or output.ndim != 2:
        raise ValueError("packed and output must be rank-2")
    if packed.shape[1] % 2 != 0:
        raise ValueError("packed last dimension must be 2 * intermediate")
    rows = int(packed.shape[0])
    intermediate = int(packed.shape[1] // 2)
    if output.shape != (rows, intermediate):
        raise ValueError(
            f"output must have shape {(rows, intermediate)}, got {tuple(output.shape)}"
        )
    if not (packed.is_contiguous() and output.is_contiguous()):
        raise ValueError("packed and output must be contiguous")
    return rows, intermediate


def swiglu_into(packed: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Fuse FP32 SiLU arithmetic + multiply over a low-precision packed gate/up tensor."""

    rows, intermediate = _validate(packed, output)
    total = rows * intermediate
    block_size = 256
    grid = (triton.cdiv(total, block_size),)
    _swiglu_kernel[grid](
        packed,
        output,
        rows,
        intermediate,
        packed.stride(0),
        output.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def swiglu(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2 or packed.shape[1] % 2 != 0:
        raise ValueError("packed must have shape [rows, 2 * intermediate]")
    output = torch.empty(
        (packed.shape[0], packed.shape[1] // 2),
        device=packed.device,
        dtype=packed.dtype,
    )
    return swiglu_into(packed, output)

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _online_softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    query_length,
    query_start,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_base = row * n_cols
    query_index = row % query_length
    query_position = query_start + query_index

    running_max = -float("inf")
    running_sum = 0.0

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        valid = cols < n_cols
        if CAUSAL:
            valid = valid & (cols <= query_position)
        values = tl.load(input_ptr + row_base + cols, mask=valid, other=-float("inf")).to(tl.float32)
        tile_max = tl.max(values, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        tile_sum = tl.sum(tl.exp(values - new_max), axis=0)
        running_sum = running_sum * tl.exp(running_max - new_max) + tile_sum
        running_max = new_max

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        cols = start + offsets
        in_bounds = cols < n_cols
        valid = in_bounds
        if CAUSAL:
            valid = valid & (cols <= query_position)
        values = tl.load(input_ptr + row_base + cols, mask=valid, other=-float("inf")).to(tl.float32)
        probabilities = tl.exp(values - running_max) / running_sum
        probabilities = tl.where(valid, probabilities, 0.0)
        tl.store(output_ptr + row_base + cols, probabilities, mask=in_bounds)


def online_softmax_into(
    scores: torch.Tensor,
    output: torch.Tensor,
    *,
    causal: bool,
    query_start: int,
    block_size: int = 256,
) -> torch.Tensor:
    if not scores.is_cuda or not output.is_cuda:
        raise ValueError("scores/output must be CUDA tensors")
    if scores.ndim != 4 or output.shape != scores.shape:
        raise ValueError("scores/output must be matching [batch, heads, query_length, key_length]")
    if scores.dtype not in (torch.float16, torch.bfloat16) or output.dtype != scores.dtype:
        raise ValueError("scores/output must have matching float16 or bfloat16 dtype")
    if not scores.is_contiguous() or not output.is_contiguous():
        raise ValueError("scores/output must be contiguous")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    batch, heads, query_length, key_length = scores.shape
    if causal and (query_start < 0 or query_start + query_length > key_length):
        raise ValueError("causal query positions must fit the key dimension")
    n_rows = batch * heads * query_length
    _online_softmax_kernel[(n_rows,)](
        output,
        scores,
        key_length,
        query_length,
        query_start,
        BLOCK_SIZE=block_size,
        CAUSAL=causal,
        num_warps=8,
    )
    return output


def online_softmax(
    scores: torch.Tensor,
    *,
    causal: bool,
    query_start: int,
    block_size: int = 256,
) -> torch.Tensor:
    output = torch.empty_like(scores)
    return online_softmax_into(
        scores,
        output,
        causal=causal,
        query_start=query_start,
        block_size=block_size,
    )

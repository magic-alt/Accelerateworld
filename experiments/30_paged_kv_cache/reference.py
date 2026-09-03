from __future__ import annotations

import torch


def allocate_page_cache(
    num_pages: int,
    kv_heads: int,
    page_size: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if min(num_pages, kv_heads, page_size, head_dim) <= 0:
        raise ValueError("paged cache dimensions must be positive")
    shape = (num_pages, kv_heads, page_size, head_dim)
    return (
        torch.zeros(shape, dtype=dtype, device=device),
        torch.zeros(shape, dtype=dtype, device=device),
    )


def block_table_tensor(
    table: list[list[int]],
    *,
    device: torch.device | str,
) -> torch.Tensor:
    if not table or not table[0]:
        raise ValueError("block table must be non-empty")
    width = len(table[0])
    if any(len(row) != width for row in table):
        raise ValueError("block table rows must have equal width")
    return torch.tensor(table, dtype=torch.int64, device=device)


def validate_positions(positions: torch.Tensor, max_tokens: int) -> None:
    if positions.ndim != 2:
        raise ValueError("positions must be [batch, tokens]")
    if positions.dtype != torch.int64:
        raise TypeError("positions must use torch.int64")
    if positions.numel() == 0:
        raise ValueError("positions must not be empty")
    minimum = int(positions.min().item())
    maximum = int(positions.max().item())
    if minimum < 0 or maximum >= max_tokens:
        raise IndexError(
            f"logical position range [{minimum}, {maximum}] exceeds max_tokens {max_tokens}"
        )


def validate_block_table(block_table: torch.Tensor, num_pages: int) -> None:
    if block_table.ndim != 2 or block_table.dtype != torch.int64:
        raise TypeError("block_table must be int64 [batch, max_blocks]")
    if block_table.numel() == 0:
        raise ValueError("block_table must not be empty")
    assigned = block_table[block_table >= 0]
    if assigned.numel() == 0:
        raise ValueError("block_table must contain at least one allocated page")
    minimum = int(assigned.min().item())
    maximum = int(assigned.max().item())
    if minimum < 0 or maximum >= num_pages:
        raise IndexError(
            f"physical page range [{minimum}, {maximum}] exceeds page pool {num_pages}"
        )


def map_positions(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    page_size: int,
    num_pages: int,
    check_bounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if block_table.ndim != 2:
        raise ValueError("block_table must be [batch, max_blocks]")
    if positions.ndim != 2 or positions.shape[0] != block_table.shape[0]:
        raise ValueError("positions batch must match block_table")
    max_tokens = int(block_table.shape[1]) * page_size
    if check_bounds:
        validate_positions(positions, max_tokens)
        validate_block_table(block_table, num_pages)

    logical_blocks = torch.div(positions, page_size, rounding_mode="floor")
    slots = positions.remainder(page_size)
    physical_pages = block_table.gather(1, logical_blocks)
    if check_bounds:
        if bool((physical_pages < 0).any().item()):
            raise IndexError("logical position maps to an unallocated page")
        if bool((physical_pages >= num_pages).any().item()):
            raise IndexError("logical position maps outside the physical page pool")
    return physical_pages, slots


def validate_contract(
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
) -> tuple[int, int, int, int, int]:
    if page_k.shape != page_v.shape or page_k.ndim != 4:
        raise ValueError("page K/V must be matching [pages, kv_heads, page_size, head_dim]")
    if new_k.shape != new_v.shape or new_k.ndim != 4:
        raise ValueError("new K/V must be matching [batch, kv_heads, tokens, head_dim]")
    num_pages, kv_heads, stored_page_size, head_dim = map(int, page_k.shape)
    batch, update_heads, tokens, update_dim = map(int, new_k.shape)
    if stored_page_size != page_size:
        raise ValueError("page_size argument must match physical page tensor")
    if update_heads != kv_heads or update_dim != head_dim:
        raise ValueError("new K/V kv_heads and head_dim must match physical pages")
    if block_table.shape[0] != batch or positions.shape != (batch, tokens):
        raise ValueError("block_table/positions batch or token count mismatch")
    if page_k.dtype != new_k.dtype or page_v.dtype != new_v.dtype:
        raise TypeError("page cache and update dtypes must match")
    map_positions(
        block_table,
        positions,
        page_size=page_size,
        num_pages=num_pages,
        check_bounds=True,
    )
    return batch, kv_heads, tokens, head_dim, num_pages


def paged_kv_update_reference(
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    check_bounds: bool = True,
) -> None:
    if check_bounds:
        validate_contract(
            page_k, page_v, new_k, new_v, positions, block_table, page_size=page_size
        )
    num_pages = int(page_k.shape[0])
    physical, slots = map_positions(
        block_table,
        positions,
        page_size=page_size,
        num_pages=num_pages,
        check_bounds=check_bounds,
    )
    batch = int(new_k.shape[0])
    for batch_index in range(batch):
        page_k[physical[batch_index], :, slots[batch_index], :] = (
            new_k[batch_index].permute(1, 0, 2)
        )
        page_v[physical[batch_index], :, slots[batch_index], :] = (
            new_v[batch_index].permute(1, 0, 2)
        )


def paged_kv_read_reference(
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    check_bounds: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if page_k.shape != page_v.shape or page_k.ndim != 4:
        raise ValueError("page K/V must be matching rank-4 tensors")
    num_pages, kv_heads, stored_page_size, head_dim = map(int, page_k.shape)
    if stored_page_size != page_size:
        raise ValueError("page_size argument must match physical page tensor")
    physical, slots = map_positions(
        block_table,
        positions,
        page_size=page_size,
        num_pages=num_pages,
        check_bounds=check_bounds,
    )
    batch, tokens = map(int, positions.shape)
    out_k = torch.empty(
        (batch, kv_heads, tokens, head_dim), dtype=page_k.dtype, device=page_k.device
    )
    out_v = torch.empty_like(out_k)
    for batch_index in range(batch):
        gathered_k = page_k[physical[batch_index], :, slots[batch_index], :]
        gathered_v = page_v[physical[batch_index], :, slots[batch_index], :]
        out_k[batch_index].copy_(gathered_k.permute(1, 0, 2))
        out_v[batch_index].copy_(gathered_v.permute(1, 0, 2))
    return out_k, out_v

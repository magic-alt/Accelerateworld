from __future__ import annotations

import torch

from reference import allocate_cache, kv_cache_read_reference, kv_cache_update_reference


def run_layout(layout: str) -> None:
    batch, kv_heads, capacity, head_dim = 2, 2, 12, 8
    cache_k, cache_v = allocate_cache(
        batch, kv_heads, capacity, head_dim, dtype=torch.float32, device="cpu", layout=layout
    )

    prefill_positions = torch.tensor([[0, 1, 2, 3], [2, 4, 6, 8]], dtype=torch.int64)
    base = torch.arange(batch * kv_heads * 4 * head_dim, dtype=torch.float32).reshape(
        batch, kv_heads, 4, head_dim
    )
    prefill_k = base / 10.0
    prefill_v = -base / 7.0
    kv_cache_update_reference(
        cache_k, cache_v, prefill_k, prefill_v, prefill_positions, layout=layout
    )
    got_k, got_v = kv_cache_read_reference(cache_k, cache_v, prefill_positions, layout=layout)
    torch.testing.assert_close(got_k, prefill_k)
    torch.testing.assert_close(got_v, prefill_v)

    decode_positions = torch.tensor([[4], [9]], dtype=torch.int64)
    decode_k = torch.full((batch, kv_heads, 1, head_dim), 3.25)
    decode_v = torch.full((batch, kv_heads, 1, head_dim), -1.75)
    kv_cache_update_reference(
        cache_k, cache_v, decode_k, decode_v, decode_positions, layout=layout
    )
    got_k, got_v = kv_cache_read_reference(cache_k, cache_v, decode_positions, layout=layout)
    torch.testing.assert_close(got_k, decode_k)
    torch.testing.assert_close(got_v, decode_v)

    try:
        bad = torch.tensor([[capacity], [0]], dtype=torch.int64)
        kv_cache_read_reference(cache_k, cache_v, bad, layout=layout)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-capacity positions must be rejected")


if __name__ == "__main__":
    run_layout("token_major")
    run_layout("head_major")
    print("KV-cache FP32 CPU reference validation: PASS")

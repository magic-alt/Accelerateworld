from __future__ import annotations

import torch
import torch.nn.functional as F

from reference import causal_mask, expand_kv_for_gqa, manual_attention_fp32


def check_case(*, causal: bool, query_start: int) -> None:
    torch.manual_seed(2026)
    q = torch.randn((1, 4, 3, 64), dtype=torch.float32)
    k = torch.randn((1, 2, 7, 64), dtype=torch.float32)
    v = torch.randn((1, 2, 7, 64), dtype=torch.float32)
    reference = manual_attention_fp32(q, k, v, causal=causal, query_start=query_start)
    k_exp, v_exp = expand_kv_for_gqa(k, v, q.shape[1])
    mask = None
    if causal:
        mask = causal_mask(q.shape[2], k.shape[2], query_start, q.device).view(1, 1, q.shape[2], k.shape[2])
    actual = F.scaled_dot_product_attention(
        q,
        k_exp,
        v_exp,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.testing.assert_close(reference, actual.float(), rtol=2e-5, atol=2e-5)


def main() -> int:
    check_case(causal=False, query_start=0)
    check_case(causal=True, query_start=4)

    # If V is all ones, every normalized attention row must return ones.
    q = torch.randn((1, 8, 1, 64), dtype=torch.float32)
    k = torch.randn((1, 2, 16, 64), dtype=torch.float32)
    v = torch.ones((1, 2, 16, 64), dtype=torch.float32)
    out = manual_attention_fp32(q, k, v, causal=True, query_start=15)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-6, atol=1e-6)
    print("FlashAttention FP32 oracle / GQA / causal normalization: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

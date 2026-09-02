from __future__ import annotations

import torch

from reference import masked_max_abs, row_sum_error, stable_softmax_fp32


def main() -> None:
    torch.manual_seed(2026)
    scores = torch.randn(1, 2, 4, 8, dtype=torch.float32) * 4.0 + 80.0
    output = stable_softmax_fp32(scores, causal=False, query_start=0)
    torch.testing.assert_close(output, torch.softmax(scores, dim=-1), rtol=1e-6, atol=1e-7)
    assert torch.isfinite(output).all()
    assert row_sum_error(output) < 1e-6

    causal_scores = torch.randn(1, 2, 4, 8, dtype=torch.float32) * 5.0 - 70.0
    causal_output = stable_softmax_fp32(causal_scores, causal=True, query_start=4)
    mask = torch.arange(8).view(1, 8) <= (torch.arange(4).view(4, 1) + 4)
    expected = torch.softmax(
        causal_scores.masked_fill(~mask.view(1, 1, 4, 8), float("-inf")), dim=-1
    )
    torch.testing.assert_close(causal_output, expected, rtol=1e-6, atol=1e-7)
    assert masked_max_abs(causal_output, causal=True, query_start=4) == 0.0
    assert row_sum_error(causal_output) < 1e-6

    print("online softmax FP32 oracle validation: PASS")
    print("  large positive/negative logit stability: PASS")
    print("  causal masking: PASS")
    print("  row normalization: PASS")


if __name__ == "__main__":
    main()

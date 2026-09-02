from __future__ import annotations

import torch

from reference import build_position_ids, build_rope_cache, rope_tensor_fp32


def _pair_norm(x: torch.Tensor, rotary_dim: int, layout: str) -> torch.Tensor:
    pair_count = rotary_dim // 2
    x = x[..., :rotary_dim]
    if layout == "interleaved":
        p = x.reshape(*x.shape[:-1], pair_count, 2)
        return p.square().sum(-1)
    return x[..., :pair_count].square() + x[..., pair_count:].square()


def main() -> int:
    torch.manual_seed(2026)
    x = torch.randn(2, 3, 4, 8)
    positions = build_position_ids(2, 3, 0, device="cpu")
    cos, sin = build_rope_cache(2, 8, device="cpu")

    for layout in ("interleaved", "half_split"):
        y = rope_tensor_fp32(x, cos, sin, positions, 8, layout)
        torch.testing.assert_close(
            _pair_norm(x, 8, layout),
            _pair_norm(y, 8, layout),
            rtol=2e-5,
            atol=2e-6,
        )

        position_zero = torch.zeros((2, 3), dtype=torch.int64)
        y0 = rope_tensor_fp32(x, cos, sin, position_zero, 8, layout)
        torch.testing.assert_close(y0, x.float(), rtol=0.0, atol=0.0)

        x_partial = torch.randn(1, 2, 3, 12)
        partial_positions = build_position_ids(1, 2, 1, device="cpu")
        cos_partial, sin_partial = build_rope_cache(2, 8, device="cpu")
        partial = rope_tensor_fp32(
            x_partial, cos_partial, sin_partial, partial_positions, 8, layout
        )
        torch.testing.assert_close(
            partial[..., 8:], x_partial[..., 8:].float(), rtol=0.0, atol=0.0
        )

    print("RoPE FP32 oracle / norm preservation / tail copy: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

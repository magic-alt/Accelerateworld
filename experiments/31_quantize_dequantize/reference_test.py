from __future__ import annotations

import torch

from codec import pack_int4_values, round_half_away_from_zero, unpack_int4_values
from lab_config import FORMATS
from reference import compute_qparams, dequantize_reference, quantize_reference, unpack_int4_tensor


def main() -> int:
    assert round_half_away_from_zero(1.5) == 2
    assert round_half_away_from_zero(-1.5) == -2
    values = [-8, -7, -1, 0, 1, 6, 7]
    payload = pack_int4_values(values)
    assert unpack_int4_values(payload, len(values)) == values
    x = torch.tensor([[-3.0, -1.25, -0.5, 0.0, 0.5, 1.25, 3.0], [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]], dtype=torch.float32)
    for format_name in FORMATS:
        for granularity, group_size in (("per_tensor", 0), ("per_channel", 0), ("group", 32)):
            scales, zero_points = compute_qparams(x, format_name=format_name, granularity=granularity, group_size=group_size)
            q = quantize_reference(x, scales, zero_points, format_name=format_name, granularity=granularity, group_size=group_size)
            dq = dequantize_reference(q, scales, zero_points, rows=x.shape[0], cols=x.shape[1], format_name=format_name, granularity=granularity, group_size=group_size, output_dtype=torch.float32)
            assert dq.shape == x.shape
            assert torch.isfinite(dq).all()
            assert float((dq - x).abs().max()) < 1.0
    scales, zero_points = compute_qparams(x, format_name="int8_asym", granularity="per_channel")
    assert torch.any(zero_points != 0), "asymmetric INT8 should exercise non-zero zero points"
    scales, zero_points = compute_qparams(x, format_name="int4_sym", granularity="per_channel")
    packed = quantize_reference(x, scales, zero_points, format_name="int4_sym", granularity="per_channel")
    unpacked = unpack_int4_tensor(packed, x.shape[1])
    assert unpacked.shape == x.shape
    assert packed.shape == (2, 4)
    print("quantize/dequantize FP32 reference validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

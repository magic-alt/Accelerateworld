from __future__ import annotations

import math


def round_half_away_from_zero(value: float) -> int:
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def clamp_int(value: int, qmin: int, qmax: int) -> int:
    return min(qmax, max(qmin, value))


def encode_int4(value: int) -> int:
    if value < -8 or value > 7:
        raise ValueError("INT4 value must be in [-8, 7]")
    return value & 0xF


def decode_int4(nibble: int) -> int:
    nibble &= 0xF
    return nibble - 16 if nibble >= 8 else nibble


def pack_int4_pair(low: int, high: int = 0) -> int:
    return encode_int4(low) | (encode_int4(high) << 4)


def unpack_int4_pair(byte: int) -> tuple[int, int]:
    return decode_int4(byte), decode_int4(byte >> 4)


def pack_int4_values(values: list[int]) -> bytes:
    out = bytearray((len(values) + 1) // 2)
    for index in range(0, len(values), 2):
        low = values[index]
        high = values[index + 1] if index + 1 < len(values) else 0
        out[index // 2] = pack_int4_pair(low, high)
    return bytes(out)


def unpack_int4_values(payload: bytes, count: int) -> list[int]:
    if count < 0 or count > len(payload) * 2:
        raise ValueError("invalid INT4 element count")
    values: list[int] = []
    for byte in payload:
        low, high = unpack_int4_pair(byte)
        values.extend((low, high))
    return values[:count]

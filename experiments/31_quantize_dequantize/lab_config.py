from __future__ import annotations

from dataclasses import dataclass


FORMAT_INFO = {
    "int8_sym": {"bits": 8, "qmin": -127, "qmax": 127, "packed": False, "asymmetric": False},
    "int8_asym": {"bits": 8, "qmin": -128, "qmax": 127, "packed": False, "asymmetric": True},
    "int4_sym": {"bits": 4, "qmin": -8, "qmax": 7, "packed": True, "asymmetric": False},
}
FORMATS = tuple(FORMAT_INFO)
GRANULARITIES = ("per_tensor", "per_channel", "group")
GRANULARITY_IDS = {"per_tensor": 0, "per_channel": 1, "group": 2}
DTYPE_REQUIREMENTS = {"fp16": ("cuda",), "bf16": ("cuda", "bf16")}


@dataclass(frozen=True)
class QuantShape:
    name: str
    family: str
    role: str
    rows: int
    cols: int

    @property
    def elements(self) -> int:
        return self.rows * self.cols

    @property
    def packed_int4_bytes(self) -> int:
        return self.rows * ((self.cols + 1) // 2)


SHAPES = (
    QuantShape("validation_odd_tail", "validation", "correctness", 7, 65),
    QuantShape("validation_grouped", "validation", "correctness", 16, 128),
    QuantShape("decode_weight_tile", "decode", "weight_tile", 256, 4096),
    QuantShape("attention_projection", "weight", "attention_weight", 4096, 4096),
    QuantShape("mlp_gate_up_projection", "weight", "mlp_weight", 4096, 11008),
)
SHAPE_FAMILIES = ("validation", "decode", "weight", "all")
GROUP_SIZES = (32, 64, 128)


def validate_shape(shape: QuantShape) -> None:
    if shape.rows <= 0 or shape.cols <= 0:
        raise ValueError(f"shape dimensions must be positive: {shape}")


def validate_shapes() -> None:
    names = set()
    for shape in SHAPES:
        validate_shape(shape)
        if shape.name in names:
            raise ValueError(f"duplicate shape name: {shape.name}")
        names.add(shape.name)
    if not any(shape.cols % 2 for shape in SHAPES):
        raise ValueError("shape matrix must include an odd INT4 tail")
    if not any(shape.rows >= 4096 and shape.cols >= 4096 for shape in SHAPES):
        raise ValueError("shape matrix must include an LLM weight-scale workload")


def shapes_for_family(family: str) -> tuple[QuantShape, ...]:
    if family not in SHAPE_FAMILIES:
        raise KeyError(f"unknown shape family: {family}")
    if family == "all":
        return SHAPES
    return tuple(shape for shape in SHAPES if shape.family == family)


def format_info(format_name: str) -> dict[str, int | bool]:
    try:
        return FORMAT_INFO[format_name]
    except KeyError as exc:
        raise KeyError(f"unsupported quantization format: {format_name}") from exc


def granularity_id(granularity: str) -> int:
    try:
        return GRANULARITY_IDS[granularity]
    except KeyError as exc:
        raise KeyError(f"unsupported granularity: {granularity}") from exc


def parameter_count(rows: int, cols: int, granularity: str, group_size: int) -> int:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if granularity == "per_tensor":
        return 1
    if granularity == "per_channel":
        return rows
    if granularity == "group":
        if group_size <= 0:
            raise ValueError("group_size must be positive for group-wise quantization")
        return rows * ((cols + group_size - 1) // group_size)
    raise KeyError(f"unsupported granularity: {granularity}")


def validate_granularity(granularity: str, group_size: int) -> None:
    if granularity not in GRANULARITIES:
        raise KeyError(f"unsupported granularity: {granularity}")
    if granularity == "group":
        if group_size not in GROUP_SIZES:
            raise ValueError(f"group_size must be one of {GROUP_SIZES}")
    elif group_size not in (0, 1):
        raise ValueError("group_size must be 0 or 1 outside group-wise mode")


def storage_bytes(format_name: str, rows: int, cols: int) -> int:
    info = format_info(format_name)
    if info["packed"]:
        return rows * ((cols + 1) // 2)
    return rows * cols


validate_shapes()

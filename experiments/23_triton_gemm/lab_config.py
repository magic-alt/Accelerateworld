from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GemmShape:
    name: str
    family: str
    role: str
    m: int
    n: int
    k: int

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


AUTOTUNE_CONFIG_SPECS: tuple[dict[str, int], ...] = (
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 3,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 2,
        "num_stages": 5,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 2,
        "num_stages": 4,
    },
)


SHAPE_FAMILIES: dict[str, tuple[GemmShape, ...]] = {
    "validation": (
        GemmShape("validation-prefill-qkv", "validation", "prefill-qkv", 128, 1536, 1024),
        GemmShape("validation-decode-proj", "validation", "decode-proj", 8, 1024, 1024),
    ),
    "balanced": (
        GemmShape("balanced-1k", "balanced", "square", 1024, 1024, 1024),
        GemmShape("balanced-2k", "balanced", "square", 2048, 2048, 2048),
    ),
    "prefill": (
        GemmShape("prefill-qkv-4k", "prefill", "qkv", 512, 12288, 4096),
        GemmShape("prefill-attn-out-4k", "prefill", "attention-output", 512, 4096, 4096),
        GemmShape("prefill-mlp-up-4k", "prefill", "mlp-up", 512, 11008, 4096),
        GemmShape("prefill-mlp-down-4k", "prefill", "mlp-down", 512, 4096, 11008),
    ),
    "decode": (
        GemmShape("decode-qkv-m1", "decode", "qkv", 1, 12288, 4096),
        GemmShape("decode-mlp-up-m1", "decode", "mlp-up", 1, 11008, 4096),
        GemmShape("decode-mlp-down-m4", "decode", "mlp-down", 4, 4096, 11008),
        GemmShape("decode-proj-m16", "decode", "projection", 16, 4096, 4096),
    ),
}


def shapes_for_family(family: str) -> tuple[GemmShape, ...]:
    if family == "all":
        return SHAPE_FAMILIES["balanced"] + SHAPE_FAMILIES["prefill"] + SHAPE_FAMILIES["decode"]
    try:
        return SHAPE_FAMILIES[family]
    except KeyError as exc:
        raise ValueError(f"unknown shape family: {family}") from exc


def validate_config_specs(specs: Iterable[dict[str, int]] = AUTOTUNE_CONFIG_SPECS) -> None:
    required = {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
    seen: set[tuple[int, ...]] = set()
    count = 0
    for spec in specs:
        count += 1
        missing = required.difference(spec)
        if missing:
            raise ValueError(f"autotune config is missing keys: {sorted(missing)}")
        signature = tuple(spec[key] for key in sorted(required))
        if signature in seen:
            raise ValueError(f"duplicate autotune config: {spec}")
        seen.add(signature)
        for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
            value = spec[key]
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{key} must be a positive power of two: {value}")
        if spec["num_warps"] not in {2, 4, 8}:
            raise ValueError(f"unsupported num_warps: {spec['num_warps']}")
        if not 2 <= spec["num_stages"] <= 5:
            raise ValueError(f"num_stages outside experiment range: {spec['num_stages']}")
    if count < 8:
        raise ValueError("autotune search space must retain at least eight configurations")


def validate_shape_families() -> None:
    required = {"validation", "balanced", "prefill", "decode"}
    if required.difference(SHAPE_FAMILIES):
        raise ValueError("missing required GEMM shape families")
    names: set[str] = set()
    for family, shapes in SHAPE_FAMILIES.items():
        if not shapes:
            raise ValueError(f"shape family {family} is empty")
        for shape in shapes:
            if shape.name in names:
                raise ValueError(f"duplicate shape name: {shape.name}")
            names.add(shape.name)
            if min(shape.m, shape.n, shape.k) <= 0:
                raise ValueError(f"invalid GEMM shape: {shape}")
    if not all(shape.m <= 16 for shape in SHAPE_FAMILIES["decode"]):
        raise ValueError("decode family must remain small-M")
    if not all(shape.m >= 128 for shape in SHAPE_FAMILIES["prefill"]):
        raise ValueError("prefill family must retain token-parallel M")


def pick_winner(latencies_ms: dict[str, float | None]) -> str:
    available = {
        provider: latency
        for provider, latency in latencies_ms.items()
        if latency is not None and latency > 0.0
    }
    if not available:
        return "unavailable"
    return min(available, key=lambda provider: (available[provider], provider))


def parse_direct_cublas_output(output: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Latency:") and stripped.endswith("ms"):
            parsed["latency_ms"] = float(stripped.split()[1])
        elif stripped.startswith("Throughput:") and stripped.endswith("GFLOP/s"):
            parsed["throughput_gflops"] = float(stripped.split()[1])
        elif stripped.startswith("Max normalized error:"):
            parsed["max_normalized_error"] = float(stripped.split(":", 1)[1].strip())
    return parsed


validate_config_specs()
validate_shape_families()

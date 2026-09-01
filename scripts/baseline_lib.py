#!/usr/bin/env python3
"""Shared helpers for GPU Baseline v2."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "hardware" / "rtx_capabilities.json"


def capture(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def detect_gpu(index: int = 0, require_rtx: bool = True) -> dict[str, Any]:
    query = capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    if query["returncode"] != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {query['stderr'] or query['stdout']}")

    selected: list[str] | None = None
    for raw_line in query["stdout"].splitlines():
        fields = [field.strip() for field in _parse_csv_line(raw_line)]
        if len(fields) < 6:
            continue
        if int(fields[0]) == index:
            selected = fields
            break
    if selected is None:
        raise RuntimeError(f"GPU index {index} was not returned by nvidia-smi")

    gpu_index, name, driver, memory_mb, compute_capability, pci_bus_id = selected[:6]
    sm = compute_capability.replace(".", "")
    sm_key = f"sm_{sm}"
    registry = load_registry()
    architecture = registry["architectures"].get(sm_key)
    is_rtx = bool(re.search(r"\bRTX\b", name, flags=re.IGNORECASE))

    if require_rtx and not is_rtx:
        raise RuntimeError(f"GPU '{name}' is not an RTX device; GPU Baseline v2 currently targets RTX GPUs")
    if architecture is None:
        supported = ", ".join(sorted(registry["architectures"].keys()))
        raise RuntimeError(
            f"Compute capability {compute_capability} ({sm_key}) is not in the RTX capability registry; "
            f"supported profiles: {supported}"
        )

    lowered_name = name.lower()
    reference = any(
        reference_name.lower() in lowered_name
        for reference_name in architecture.get("reference_gpus", [])
    )
    return {
        "index": int(gpu_index),
        "name": name,
        "driver_version": driver,
        "memory_mb": int(float(memory_mb)),
        "compute_capability": compute_capability,
        "sm": sm,
        "sm_key": sm_key,
        "pci_bus_id": pci_bus_id,
        "is_rtx": is_rtx,
        "generation": architecture["generation"],
        "rtx_series": architecture["rtx_series"],
        "features": architecture["features"],
        "reference_gpu": reference,
    }


def missing_features(gpu: dict[str, Any], required: list[str]) -> list[str]:
    features = gpu.get("features", {})
    return [feature for feature in required if not features.get(feature, False)]


_METRIC_LINE = re.compile(
    r"^\s*([^:]+):\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([^,\s]+)?"
    r"(?:,\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(.+?))?\s*$"
)


def parse_metrics(stdout: str) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        label, first_value, first_unit, second_value, second_unit = match.groups()
        key = normalize_key(label)
        if not key:
            continue
        metrics[key] = {
            "value": float(first_value),
            "unit": (first_unit or "").strip(),
        }
        if second_value is not None:
            metrics[f"{key}_secondary"] = {
                "value": float(second_value),
                "unit": (second_unit or "").strip(),
            }
    return metrics


def slugify_gpu(name: str) -> str:
    return normalize_key(name).replace("_", "-")

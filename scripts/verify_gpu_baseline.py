#!/usr/bin/env python3
"""Verify an Accelerateworld GPU Baseline v2 report without rerunning the GPU workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path, help="Path to GPU Baseline v2 baseline.json")
    parser.add_argument("--expected-gpu", help="Optional case-insensitive substring expected in GPU name")
    parser.add_argument("--expected-cc", help="Optional expected compute capability, e.g. 12.0")
    parser.add_argument("--expected-series", choices=["RTX 20", "RTX 30", "RTX 40", "RTX 50"])
    args = parser.parse_args()

    errors: list[str] = []
    try:
        payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"GPU Baseline v2 evidence: FAIL\n  - cannot read report: {error}")
        return 2

    if payload.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if payload.get("kind") != "accelerateworld-gpu-baseline":
        errors.append("kind must be accelerateworld-gpu-baseline")

    gpu = payload.get("gpu", {})
    if not gpu.get("is_rtx"):
        errors.append("report does not identify an RTX GPU")
    if args.expected_gpu and args.expected_gpu.lower() not in str(gpu.get("name", "")).lower():
        errors.append(f"GPU name does not contain {args.expected_gpu!r}")
    if args.expected_cc and gpu.get("compute_capability") != args.expected_cc:
        errors.append(
            f"compute capability is {gpu.get('compute_capability')!r}, expected {args.expected_cc!r}"
        )
    if args.expected_series and gpu.get("rtx_series") != args.expected_series:
        errors.append(f"RTX series is {gpu.get('rtx_series')!r}, expected {args.expected_series!r}")

    validation_steps = payload.get("validation", {}).get("steps", [])
    failed_steps = [step.get("name", "unknown") for step in validation_steps if step.get("status") == "failed"]
    if failed_steps:
        errors.append("validation failures: " + ", ".join(failed_steps))

    benchmarks = payload.get("benchmarks", [])
    if not benchmarks:
        errors.append("report has no benchmark records")
    failed_benchmarks = [item.get("id", "unknown") for item in benchmarks if item.get("status") == "failed"]
    if failed_benchmarks:
        errors.append("benchmark failures: " + ", ".join(failed_benchmarks))

    if not payload.get("success"):
        errors.append("top-level success is false")

    if errors:
        print("GPU Baseline v2 evidence: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 2

    print("GPU Baseline v2 evidence: PASS")
    print(f"  GPU: {gpu.get('name')}")
    print(f"  Generation: {gpu.get('generation')} / {gpu.get('rtx_series')}")
    print(f"  Compute capability: {gpu.get('compute_capability')} / sm_{gpu.get('sm')}")
    print(f"  Reference GPU: {gpu.get('reference_gpu')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

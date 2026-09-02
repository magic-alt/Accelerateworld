#!/usr/bin/env python3
"""Run the canonical native-CUDA benchmark suite with GPU Baseline v2 feature gates."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from baseline_lib import capture, detect_gpu, missing_features, parse_metrics


def collect_environment(gpu: dict[str, Any]) -> dict[str, Any]:
    nvcc = capture(["nvcc", "--version"])
    git = capture(["git", "rev-parse", "HEAD"])
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git["stdout"] or None,
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": gpu,
        "nvcc": nvcc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/manifest.json")
    parser.add_argument("--build-dir", default="build/gpu")
    parser.add_argument("--output", default="results/native-benchmark.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--allow-non-rtx", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="Run only the named benchmark id")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    gpu = detect_gpu(args.gpu_index, require_rtx=not args.allow_non_rtx)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(args.only)
    build_dir = str(Path(args.build_dir))
    exe = ".exe" if os.name == "nt" else ""
    results: list[dict[str, Any]] = []
    failed = False

    for item in manifest["benchmarks"]:
        if selected and item["id"] not in selected:
            continue

        required = item.get("requires", [])
        missing = missing_features(gpu, required)
        if missing:
            print(f"== {item['id']} == SKIPPED (missing: {', '.join(missing)})", flush=True)
            results.append(
                {
                    "id": item["id"],
                    "description": item["description"],
                    "status": "skipped",
                    "requires": required,
                    "missing_features": missing,
                    "primary_metric": item.get("primary_metric"),
                    "metrics": {},
                }
            )
            continue

        command = [
            part.format(build_dir=build_dir, exe=exe)
            for part in item["command"]
        ]
        print(f"== {item['id']} ==", flush=True)
        print(" ".join(command), flush=True)
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
        elapsed = time.perf_counter() - start
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        status = "passed" if completed.returncode == 0 else "failed"
        failed = failed or status == "failed"

        results.append(
            {
                "id": item["id"],
                "description": item["description"],
                "status": status,
                "requires": required,
                "primary_metric": item.get("primary_metric"),
                "command": command,
                "returncode": completed.returncode,
                "host_elapsed_seconds": elapsed,
                "metrics": parse_metrics(completed.stdout),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )

    report = {
        "schema_version": 2,
        "kind": "accelerateworld-benchmark-suite",
        "suite": manifest["suite"],
        "environment": collect_environment(gpu),
        "benchmarks": results,
        "success": not failed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Benchmark report: {output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

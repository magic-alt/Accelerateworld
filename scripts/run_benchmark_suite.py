#!/usr/bin/env python3
"""Run the canonical native-CUDA benchmark suite and preserve reproducible evidence."""

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


def collect_environment() -> dict[str, Any]:
    gpu = capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    nvcc = capture(["nvcc", "--version"])
    git = capture(["git", "rev-parse", "HEAD"])
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git["stdout"] or None,
        "platform": platform.platform(),
        "python": sys.version,
        "gpu_query": gpu,
        "nvcc": nvcc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/manifest.json")
    parser.add_argument("--build-dir", default="build/gpu")
    parser.add_argument("--output", default="results/native-benchmark.json")
    parser.add_argument("--only", action="append", default=[], help="Run only the named benchmark id")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(args.only)
    build_dir = str(Path(args.build_dir))
    results: list[dict[str, Any]] = []
    failed = False

    for item in manifest["benchmarks"]:
        if selected and item["id"] not in selected:
            continue
        command = [part.format(build_dir=build_dir) for part in item["command"]]
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
        if completed.returncode != 0:
            failed = True

        results.append(
            {
                "id": item["id"],
                "description": item["description"],
                "command": command,
                "returncode": completed.returncode,
                "host_elapsed_seconds": elapsed,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )

    report = {
        "schema_version": manifest["schema_version"],
        "suite": manifest["suite"],
        "environment": collect_environment(),
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

#!/usr/bin/env python3
"""GPU Baseline v2: auto-detect RTX architecture, validate, benchmark and retain evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from baseline_lib import capture, detect_gpu, missing_features, slugify_gpu


def run_step(name: str, command: list[str], timeout: int) -> dict[str, Any]:
    print(f"\n== {name} ==")
    print(" ".join(command))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        return {
            "name": name,
            "command": command,
            "returncode": completed.returncode,
            "status": "passed" if completed.returncode == 0 else "failed",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(f"{name}: {error}", file=sys.stderr)
        return {
            "name": name,
            "command": command,
            "returncode": None,
            "status": "failed",
            "stdout": "",
            "stderr": str(error),
        }


def sanitizer_steps(build_dir: Path, exe: str, gpu: dict[str, Any]) -> list[tuple[str, list[str], list[str]]]:
    candidates = [
        (
            "sanitizer-memcheck-reduction",
            ["cuda"],
            ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", str(build_dir / "bin" / f"aw_reduction{exe}"), "--elements", "262144", "--iterations", "1"],
        ),
        (
            "sanitizer-memcheck-prefix-scan",
            ["cuda"],
            ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99", str(build_dir / "bin" / f"aw_prefix_scan{exe}"), "--elements", "65536", "--iterations", "1"],
        ),
        (
            "sanitizer-racecheck-prefix-scan",
            ["cuda"],
            ["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "99", str(build_dir / "bin" / f"aw_prefix_scan{exe}"), "--elements", "4096", "--iterations", "1"],
        ),
        (
            "sanitizer-racecheck-transpose",
            ["cuda"],
            ["compute-sanitizer", "--tool", "racecheck", "--error-exitcode", "99", str(build_dir / "bin" / f"aw_transpose{exe}"), "--size", "128", "--iterations", "1"],
        ),
        (
            "sanitizer-synccheck-wmma",
            ["tensor_core", "wmma_fp16"],
            ["compute-sanitizer", "--tool", "synccheck", "--error-exitcode", "99", str(build_dir / "bin" / f"aw_tensor_core_wmma{exe}"), "--size", "64", "--iterations", "1"],
        ),
    ]
    output: list[tuple[str, list[str], list[str]]] = []
    for name, required, command in candidates:
        if not missing_features(gpu, required):
            output.append((name, required, command))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--build-dir", default="build/gpu-native")
    parser.add_argument("--result-root", default="results/gpu-baselines")
    parser.add_argument("--skip-sanitizer", action="store_true")
    parser.add_argument("--allow-non-rtx", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    gpu = detect_gpu(args.gpu_index, require_rtx=not args.allow_non_rtx)
    commit = capture(["git", "rev-parse", "HEAD"])["stdout"] or "unknown"
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = Path(args.result_root) / slugify_gpu(gpu["name"]) / f"{timestamp}-{commit[:8]}"
    result_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.build_dir)
    exe = ".exe" if os.name == "nt" else ""

    print("Accelerateworld GPU Baseline v2")
    print(f"GPU: {gpu['name']}")
    print(f"Generation: {gpu['generation']} ({gpu['rtx_series']})")
    print(f"Compute capability: {gpu['compute_capability']} / sm_{gpu['sm']}")
    print(f"Reference GPU: {gpu['reference_gpu']}")

    steps: list[dict[str, Any]] = []
    configure = [
        "cmake", "-S", ".", "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DACCELERATEWORLD_CUDA_ARCHITECTURES={gpu['sm']}",
    ]
    steps.append(run_step("configure", configure, args.timeout))
    if steps[-1]["status"] == "passed":
        steps.append(run_step("build", ["cmake", "--build", str(build_dir), "--parallel"], args.timeout))
    if all(step["status"] == "passed" for step in steps):
        steps.append(run_step("ctest", ["ctest", "--test-dir", str(build_dir), "--output-on-failure"], args.timeout))

    if not args.skip_sanitizer and all(step["status"] == "passed" for step in steps):
        for name, required, command in sanitizer_steps(build_dir, exe, gpu):
            step = run_step(name, command, args.timeout)
            step["requires"] = required
            steps.append(step)
            if step["status"] != "passed":
                break

    suite_path = result_dir / "native-benchmark.json"
    if all(step["status"] == "passed" for step in steps):
        benchmark_step = run_step(
            "benchmark-suite",
            [
                sys.executable,
                "scripts/run_benchmark_suite.py",
                "--build-dir", str(build_dir),
                "--gpu-index", str(args.gpu_index),
                "--output", str(suite_path),
            ] + (["--allow-non-rtx"] if args.allow_non_rtx else []),
            args.timeout,
        )
        steps.append(benchmark_step)

    benchmark_report: dict[str, Any] | None = None
    if suite_path.exists():
        benchmark_report = json.loads(suite_path.read_text(encoding="utf-8"))

    success = all(step["status"] == "passed" for step in steps) and bool(
        benchmark_report and benchmark_report.get("success")
    )
    report = {
        "schema_version": 2,
        "kind": "accelerateworld-gpu-baseline",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": {"commit": commit},
        "gpu": gpu,
        "software": {
            "nvidia_smi": capture(["nvidia-smi"]),
            "nvcc": capture(["nvcc", "--version"]),
            "cmake": capture(["cmake", "--version"]),
            "python": sys.version,
        },
        "validation": {"steps": steps},
        "benchmarks": benchmark_report["benchmarks"] if benchmark_report else [],
        "success": success,
    }
    baseline_path = result_dir / "baseline.json"
    baseline_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (result_dir / "gpu-profile.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")
    print(f"\nBaseline evidence: {baseline_path}")
    print("GPU Baseline v2: PASS" if success else "GPU Baseline v2: FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

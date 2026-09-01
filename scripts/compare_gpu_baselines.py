#!/usr/bin/env python3
"""Compare multiple GPU Baseline v2 reports across RTX generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") != "accelerateworld-gpu-baseline" or data.get("schema_version") != 2:
        raise ValueError(f"{path} is not an Accelerateworld GPU Baseline v2 report")
    return data


def primary_value(benchmark: dict[str, Any]) -> tuple[str, str]:
    primary = benchmark.get("primary_metric") or {}
    key = primary.get("key")
    metric = (benchmark.get("metrics") or {}).get(key) if key else None
    if not metric:
        return benchmark.get("status", "unknown"), ""
    value = metric.get("value")
    unit = metric.get("unit", "")
    return f"{value:g}", unit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", dest="json_output", type=Path)
    args = parser.parse_args()

    reports = [load_report(path) for path in args.reports]
    benchmark_ids = sorted(
        {benchmark["id"] for report in reports for benchmark in report.get("benchmarks", [])}
    )
    by_gpu: list[dict[str, Any]] = []
    for report in reports:
        gpu = report["gpu"]
        lookup = {benchmark["id"]: benchmark for benchmark in report.get("benchmarks", [])}
        by_gpu.append({"report": report, "gpu": gpu, "benchmarks": lookup})

    headers = ["Benchmark"] + [f"{item['gpu']['name']} ({item['gpu']['generation']})" for item in by_gpu]
    rows: list[list[str]] = []
    for benchmark_id in benchmark_ids:
        row = [benchmark_id]
        for item in by_gpu:
            benchmark = item["benchmarks"].get(benchmark_id)
            if not benchmark:
                row.append("n/a")
                continue
            value, unit = primary_value(benchmark)
            row.append(f"{value} {unit}".strip())
        rows.append(row)

    markdown_lines = [
        "# Accelerateworld GPU Baseline v2 comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    markdown = "\n".join(markdown_lines) + "\n"
    print(markdown)

    comparison = {
        "schema_version": 2,
        "kind": "accelerateworld-gpu-comparison",
        "gpus": [item["gpu"] for item in by_gpu],
        "benchmarks": [
            {
                "id": row[0],
                "values": {headers[index]: row[index] for index in range(1, len(headers))},
            }
            for row in rows
        ],
    }
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

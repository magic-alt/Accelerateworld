from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

from trace_config import STATS_REPORTS, build_profile_command, build_stats_command


def _capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _nsys_supports_pytorch_annotations(nsys: str) -> bool:
    help_result = _capture([nsys, "profile", "--help"])
    combined = f"{help_result.stdout}\n{help_result.stderr}"
    return help_result.returncode == 0 and "--pytorch" in combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/nsys-framework-trace"))
    parser.add_argument("--mode", choices=("steady", "full"), default="steady")
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--disable-pytorch-annotations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    workload = repo_root / "experiments" / "24_nsys_trace" / "trace_workload.py"
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_base = output_dir / f"framework-trace-{args.mode}"

    if args.dry_run:
        command = build_profile_command(
            nsys=args.nsys,
            python_exe=args.python_exe,
            workload=workload,
            output_base=output_base,
            mode=args.mode,
            elements=args.elements,
            m=args.m,
            n=args.n,
            k=args.k,
            warmup=args.warmup,
            iterations=args.iterations,
            enable_pytorch_annotations=not args.disable_pytorch_annotations,
        )
        print(
            json.dumps(
                {"profile_command": command, "stats_reports": list(STATS_REPORTS)},
                indent=2,
            )
        )
        return 0

    resolved_nsys = shutil.which(args.nsys) if not Path(args.nsys).is_file() else args.nsys
    if resolved_nsys is None:
        raise RuntimeError("Nsight Systems CLI 'nsys' was not found on PATH")
    nsys = str(resolved_nsys)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(f"{output_base}.nsys-rep")
    sqlite_path = Path(f"{output_base}.sqlite")
    if sqlite_path.exists():
        sqlite_path.unlink()

    pytorch_annotations = (
        not args.disable_pytorch_annotations and _nsys_supports_pytorch_annotations(nsys)
    )
    profile_command = build_profile_command(
        nsys=nsys,
        python_exe=args.python_exe,
        workload=workload,
        output_base=output_base,
        mode=args.mode,
        elements=args.elements,
        m=args.m,
        n=args.n,
        k=args.k,
        warmup=args.warmup,
        iterations=args.iterations,
        enable_pytorch_annotations=pytorch_annotations,
    )

    version_result = _capture([nsys, "--version"])
    print("== Nsight Systems framework trace ==")
    print(" ".join(profile_command))
    profile = _capture(profile_command)
    (output_dir / "profile.stdout.txt").write_text(profile.stdout, encoding="utf-8")
    (output_dir / "profile.stderr.txt").write_text(profile.stderr, encoding="utf-8")
    if profile.stdout:
        print(profile.stdout, end="")
    if profile.stderr:
        print(profile.stderr, file=sys.stderr, end="")
    if profile.returncode != 0:
        raise RuntimeError(f"nsys profile failed with exit code {profile.returncode}")
    if not report_path.exists():
        raise RuntimeError(f"Nsight Systems report was not produced: {report_path}")

    stats_files: dict[str, str] = {}
    stats_failures: list[str] = []
    for report in STATS_REPORTS:
        command = build_stats_command(nsys=nsys, report=report, nsys_report=report_path)
        result = _capture(command)
        output_path = output_dir / f"{report}.txt"
        output_path.write_text(
            result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )
        stats_files[report] = str(output_path)
        if result.returncode != 0:
            stats_failures.append(report)

    metadata = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": args.mode,
        "nsys_version": (version_result.stdout or version_result.stderr).strip(),
        "pytorch_annotations_enabled": pytorch_annotations,
        "profile_command": profile_command,
        "report": str(report_path),
        "report_bytes": report_path.stat().st_size,
        "stats_reports": stats_files,
        "stats_failures": stats_failures,
        "workload": {
            "elements": args.elements,
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
    }
    (output_dir / "trace-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if stats_failures:
        raise RuntimeError(f"nsys stats failed for reports: {', '.join(stats_failures)}")

    print(f"Nsight report: {report_path}")
    print(f"PyTorch annotations: {'enabled' if pytorch_annotations else 'not available/disabled'}")
    print(f"Stats reports: {', '.join(STATS_REPORTS)}")
    print("Validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

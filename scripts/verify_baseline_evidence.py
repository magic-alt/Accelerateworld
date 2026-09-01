#!/usr/bin/env python3
"""Verify the evidence bundle produced by an RTX 5060 baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TEXT = {
    "gpu-validation.log": ["GPU validation: PASS"],
    "rtx5060-device-query.txt": ["Compute capability: 12.0", "RTX 5060"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--require-ai", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    errors: list[str] = []

    for filename, needles in REQUIRED_TEXT.items():
        path = result_dir / filename
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            if needle not in text:
                errors.append(f"{path} does not contain required marker: {needle!r}")

    native_json = result_dir / "rtx5060-native.json"
    if not native_json.exists():
        errors.append(f"missing {native_json}")
    else:
        try:
            payload = json.loads(native_json.read_text(encoding="utf-8"))
            benchmarks = payload.get("benchmarks", [])
            if not benchmarks:
                errors.append(f"{native_json} has no benchmark records")
            failed = [item.get("id", "unknown") for item in benchmarks if item.get("returncode") != 0]
            if failed:
                errors.append(f"native benchmarks failed: {', '.join(failed)}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot parse {native_json}: {exc}")

    if args.require_ai:
        ai_log = result_dir / "ai-gpu-validation.log"
        if not ai_log.exists():
            errors.append(f"missing {ai_log}")
        else:
            text = ai_log.read_text(encoding="utf-8", errors="replace")
            if "AI/GPU validation: PASS" not in text:
                errors.append(f"{ai_log} does not contain AI/GPU validation PASS marker")

    if errors:
        print("RTX 5060 baseline evidence: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 2

    print("RTX 5060 baseline evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

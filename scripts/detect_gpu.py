#!/usr/bin/env python3
"""Detect an RTX GPU and resolve its Accelerateworld capability profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline_lib import detect_gpu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-non-rtx", action="store_true")
    args = parser.parse_args()

    gpu = detect_gpu(args.gpu_index, require_rtx=not args.allow_non_rtx)
    payload = json.dumps(gpu, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

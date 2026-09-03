#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
INDEX = EXPERIMENTS / "README.md"
DIR_RE = re.compile(r"^\d{2}_.+")
MIN_CHARS = 1200
MIN_SECTIONS = 5


def main() -> int:
    directories = sorted(
        path for path in EXPERIMENTS.iterdir() if path.is_dir() and DIR_RE.match(path.name)
    )
    if not directories:
        raise SystemExit("no numbered experiment directories found")
    if not INDEX.is_file():
        raise SystemExit("experiments/README.md is required")
    index_text = INDEX.read_text(encoding="utf-8")
    failures: list[str] = []
    for directory in directories:
        readme = directory / "README.md"
        if not readme.is_file():
            failures.append(f"{directory.name}: missing README.md")
            continue
        text = readme.read_text(encoding="utf-8")
        if len(text.strip()) < MIN_CHARS:
            failures.append(f"{directory.name}: README too short ({len(text.strip())} < {MIN_CHARS})")
        sections = sum(1 for line in text.splitlines() if line.startswith("## "))
        if sections < MIN_SECTIONS:
            failures.append(f"{directory.name}: expected >= {MIN_SECTIONS} level-2 sections, found {sections}")
        if "```" not in text:
            failures.append(f"{directory.name}: README needs at least one code/example block")
        if directory.name not in index_text:
            failures.append(f"{directory.name}: missing from experiments/README.md index")
    if failures:
        raise SystemExit("Experiment documentation validation failed:\n- " + "\n- ".join(failures))
    print(f"Experiment documentation validation: PASS ({len(directories)} tutorials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

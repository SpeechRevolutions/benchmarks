"""
Report writers: JSON, Markdown, CSV, and a console summary.

A benchmark's ``score()`` returns a results dict shaped::

    {
      "benchmark": "wer",
      "provider": "speech_revolutions",
      "suite_version": "v1",
      "summary": {<spec output fields>},   # headline metrics
      "per_file": [ {...}, ... ],           # optional, drives CSV
    }

Every benchmark produces the same four output formats from this single dict.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def write_markdown(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    bench = results.get("benchmark", "?")
    provider = results.get("provider", "?")
    ver = results.get("suite_version", "?")
    lines.append(f"# {bench} — {provider} ({ver})")
    lines.append("")

    summary = results.get("summary", {})
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in summary.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines.append("")

    per_file = results.get("per_file") or []
    if per_file:
        lines.append(f"_{len(per_file)} files scored. See CSV for per-file detail._")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_csv(results: dict, path: Path) -> None:
    per_file = results.get("per_file") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    if not per_file:
        # still emit the summary as a single-row CSV
        summary = results.get("summary", {})
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(summary.keys()))
            w.writerow([_csv_val(v) for v in summary.values()])
        return
    # union of keys across rows, stable order from the first row
    keys: list[str] = []
    for row in per_file:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in per_file:
            w.writerow({k: _csv_val(row.get(k)) for k in keys})


def print_console(results: dict, regressions: list[str] | None = None) -> None:
    bench = results.get("benchmark", "?")
    provider = results.get("provider", "?")
    print(f"\n=== {bench} | {provider} | {results.get('suite_version', '?')} ===")
    for k, v in results.get("summary", {}).items():
        print(f"  {k:32} {_fmt(v)}")
    if regressions:
        print("\n  REGRESSIONS:")
        for r in regressions:
            print(f"    FAIL  {r}")
    elif regressions is not None:
        print("\n  baseline check: PASS")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _csv_val(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v

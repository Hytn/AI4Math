#!/usr/bin/env python3
"""Summarize outputs produced by ``scripts/eval/run_profile_matrix.py``."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_eval_files(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("*/*/evals/eval_*.json"))


def collect_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_file in _find_eval_files(output_root):
        profile = eval_file.parents[1].name
        benchmark_dir = eval_file.parents[2].name
        payload = _read_json(eval_file)
        metrics = payload.get("metrics", {}) or {}
        row = {
            "benchmark": payload.get("benchmark", benchmark_dir),
            "split": payload.get("split", ""),
            "profile": profile,
            "provider": payload.get("provider", ""),
            "model": payload.get("model", ""),
            "lean_mode": payload.get("lean_mode", ""),
            "verification": payload.get("verification", ""),
            "max_samples": payload.get("max_samples", ""),
            "elapsed_s": payload.get("elapsed_s", ""),
            "total": metrics.get("total", 0),
            "solved": metrics.get("solved", 0),
            "solve_rate": metrics.get("solve_rate", 0),
            "avg_attempts": metrics.get("avg_attempts", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "eval_file": str(eval_file),
        }
        for key, value in sorted(metrics.items()):
            if isinstance(key, str) and key.startswith("pass@"):
                row[key] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "split",
        "profile",
        "provider",
        "model",
        "lean_mode",
        "verification",
        "max_samples",
        "total",
        "solved",
        "solve_rate",
    ]
    pass_keys = sorted({k for row in rows for k in row if k.startswith("pass@")},
                       key=lambda x: int(x.split("@", 1)[1]))
    fieldnames.extend(pass_keys)
    fieldnames.extend(["avg_attempts", "total_tokens", "elapsed_s", "eval_file"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pass_keys = sorted({k for row in rows for k in row if k.startswith("pass@")},
                       key=lambda x: int(x.split("@", 1)[1]))
    cols = ["benchmark", "profile", "total", "solved", "solve_rate", *pass_keys]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize profile matrix results")
    parser.add_argument("output_root", help="Matrix output root directory")
    parser.add_argument("--csv", default=None, help="CSV output path")
    parser.add_argument("--json", default=None, help="JSON output path")
    parser.add_argument("--md", default=None, help="Markdown table output path")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    rows = collect_rows(output_root)
    if not rows:
        raise SystemExit(f"no eval files found under {output_root}")

    csv_path = Path(args.csv) if args.csv else output_root / "summary.csv"
    json_path = Path(args.json) if args.json else output_root / "summary.json"
    md_path = Path(args.md) if args.md else output_root / "summary.md"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, rows)

    print(f"rows: {len(rows)}")
    print(f"csv : {csv_path}")
    print(f"json: {json_path}")
    print(f"md  : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

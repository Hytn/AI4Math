#!/usr/bin/env python3
"""Run a profile x benchmark evaluation matrix.

This script is intentionally a thin scheduler around ``run_eval.py``.  The
single-run evaluation semantics stay in one place; this layer only expands a
YAML matrix, chooses per-combination output directories, and records run status.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


_BOOL_FLAGS = {
    "resume": "--resume",
    "no_knowledge": "--no-knowledge",
    "cache": "--cache",
    "cache_all": "--cache-all",
    "policy_engine": "--policy-engine",
    "omit_temperature": "--omit-temperature",
}

_VALUE_FLAGS = {
    "provider": "--provider",
    "model": "--model",
    "max_samples": "--max-samples",
    "lean_mode": "--lean-mode",
    "backend": "--backend",
    "backend_url": "--backend-url",
    "backend_api_key": "--backend-api-key",
    "world_model": "--world-model",
    "dialog_index": "--dialog-index",
    "knowledge_db": "--knowledge-db",
    "plugins_dir": "--plugins-dir",
    "lemma_bank_db": "--lemma-bank-db",
    "lean_version": "--lean-version",
    "mathlib_rev": "--mathlib-rev",
    "api_base": "--api-base",
    "pool_size": "--pool-size",
    "temperature": "--temperature",
    "max_turns": "--max-turns",
    "profile_timeout": "--profile-timeout",
    "max_total_tokens": "--max-total-tokens",
}

_BENCHMARK_PROJECT_DIRS = {
    "minif2f": "data/miniF2F",
    "proofnet": "data/ProofNet",
    "putnambench": "data/PutnamBench/lean4",
    "putnam": "data/PutnamBench/lean4",
    "builtin": ".",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    split: str = "test"
    project_dir: str | None = None
    limit: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatrixRun:
    benchmark: BenchmarkSpec
    profile: ProfileSpec
    output_dir: Path
    log_file: Path
    command: list[str]


def slug(value: str) -> str:
    """Return a filesystem-friendly id without hiding useful names."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return out.strip("._") or "unnamed"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"matrix config must be a mapping: {path}")
    return data


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _read_benchmark(item: str | dict[str, Any], default_split: str) -> BenchmarkSpec:
    if isinstance(item, str):
        name = item
        raw: dict[str, Any] = {}
    elif isinstance(item, dict):
        raw = dict(item)
        name = raw.pop("name", None)
    else:
        raise TypeError(f"benchmark entries must be strings or mappings: {item!r}")
    if not name:
        raise ValueError(f"benchmark entry is missing name: {item!r}")

    split = str(raw.pop("split", default_split))
    project_dir = raw.pop("project_dir", None)
    limit = raw.pop("limit", None)
    return BenchmarkSpec(
        name=str(name),
        split=split,
        project_dir=str(project_dir) if project_dir else None,
        limit=int(limit) if limit is not None else None,
        options=raw,
    )


def _read_profile(item: str | dict[str, Any]) -> ProfileSpec:
    if isinstance(item, str):
        return ProfileSpec(name=item)
    if not isinstance(item, dict):
        raise TypeError(f"profile entries must be strings or mappings: {item!r}")
    raw = dict(item)
    name = raw.pop("name", None)
    if not name:
        raise ValueError(f"profile entry is missing name: {item!r}")
    return ProfileSpec(name=str(name), options=raw)


def parse_matrix(config: dict[str, Any]) -> tuple[list[BenchmarkSpec], list[ProfileSpec]]:
    default_split = str(config.get("split", "test"))
    benchmarks = [_read_benchmark(x, default_split) for x in config.get("benchmarks", [])]
    profiles = [_read_profile(x) for x in config.get("profiles", [])]
    if not benchmarks:
        raise ValueError("matrix config must define at least one benchmark")
    if not profiles:
        raise ValueError("matrix config must define at least one profile")
    return benchmarks, profiles


def _merged_options(
    config: dict[str, Any],
    benchmark: BenchmarkSpec,
    profile: ProfileSpec,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    options.update(config.get("defaults", {}) or {})

    # Keep backward-compatible top-level keys for small configs.
    for key in [*_VALUE_FLAGS, *_BOOL_FLAGS, "limit", "samples"]:
        if key in config:
            options[key] = config[key]

    options.update(benchmark.options)
    if benchmark.limit is not None:
        options["limit"] = benchmark.limit
    options.update(profile.options)

    if "samples" in options and "max_samples" not in options:
        options["max_samples"] = options.pop("samples")
    return options


def _append_run_eval_options(
    cmd: list[str],
    options: dict[str, Any],
    benchmark: BenchmarkSpec,
) -> None:
    for key, flag in _BOOL_FLAGS.items():
        if _as_bool(options.get(key, False)):
            cmd.append(flag)

    for key, flag in _VALUE_FLAGS.items():
        value = options.get(key)
        if value is not None and value != "":
            cmd.extend([flag, str(value)])

    limit = options.get("limit")
    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])

    project_dir = benchmark.project_dir
    if project_dir is None:
        project_dir = _BENCHMARK_PROJECT_DIRS.get(benchmark.name.lower())
    if project_dir:
        cmd.extend(["--project-dir", project_dir])


def build_runs(
    config: dict[str, Any],
    *,
    output_root: Path | None = None,
    python_exe: str | None = None,
    benchmark_filter: set[str] | None = None,
    profile_filter: set[str] | None = None,
) -> list[MatrixRun]:
    benchmarks, profiles = parse_matrix(config)
    python_exe = python_exe or sys.executable

    if output_root is None:
        configured = config.get("output_dir")
        if configured:
            output_root = Path(str(configured))
        else:
            stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            output_root = Path("results") / "profile_matrix" / stamp
    output_root = output_root if output_root.is_absolute() else REPO_ROOT / output_root

    runs: list[MatrixRun] = []
    for benchmark in benchmarks:
        if benchmark_filter and benchmark.name not in benchmark_filter:
            continue
        for profile in profiles:
            if profile_filter and profile.name not in profile_filter:
                continue
            run_output = output_root / slug(benchmark.name) / slug(profile.name)
            log_file = output_root / "logs" / slug(benchmark.name) / f"{slug(profile.name)}.log"
            options = _merged_options(config, benchmark, profile)
            cmd = [
                python_exe,
                str(REPO_ROOT / "run_eval.py"),
                "--benchmark",
                benchmark.name,
                "--split",
                benchmark.split,
                "--profile",
                profile.name,
                "--output-dir",
                str(run_output),
            ]
            _append_run_eval_options(cmd, options, benchmark)
            runs.append(MatrixRun(benchmark, profile, run_output, log_file, cmd))
    return runs


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_command(run: MatrixRun, *, dry_run: bool = False) -> dict[str, Any]:
    started = time.time()
    run.log_file.parent.mkdir(parents=True, exist_ok=True)
    run.output_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "benchmark": run.benchmark.name,
        "split": run.benchmark.split,
        "profile": run.profile.name,
        "output_dir": str(run.output_dir),
        "log_file": str(run.log_file),
        "command": run.command,
        "returncode": None,
        "elapsed_s": 0.0,
        "status": "dry_run" if dry_run else "running",
    }
    if dry_run:
        print(" ".join(run.command))
        return record

    with run.log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(run.command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            run.command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        returncode = proc.wait()

    record["returncode"] = returncode
    record["elapsed_s"] = round(time.time() - started, 1)
    record["status"] = "ok" if returncode == 0 else "failed"
    return record


def run_matrix(
    runs: list[MatrixRun],
    *,
    output_root: Path,
    config_path: Path,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for idx, run in enumerate(runs, 1):
        print(
            f"\n[{idx}/{len(runs)}] "
            f"{run.benchmark.name}/{run.profile.name} -> {run.output_dir}"
        )
        record = _run_command(run, dry_run=dry_run)
        records.append(record)
        _write_json(output_root / "matrix_runs.json", records)
        if fail_fast and record["status"] == "failed":
            break

    manifest = {
        "config": str(config_path),
        "output_root": str(output_root),
        "total": len(records),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "dry_run": dry_run,
        "runs_file": str(output_root / "matrix_runs.json"),
    }
    _write_json(output_root / "matrix_manifest.json", manifest)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run profile x benchmark eval matrix")
    parser.add_argument("--config", required=True, help="YAML matrix config")
    parser.add_argument("--output-dir", default=None, help="Override matrix output root")
    parser.add_argument("--benchmark", action="append", default=None,
                        help="Only run this benchmark name; repeatable")
    parser.add_argument("--profile", action="append", default=None,
                        help="Only run this profile name; repeatable")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print expanded commands without running them")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first failed combination")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to invoke run_eval.py")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)

    output_setting = args.output_dir or config.get("output_dir")
    if output_setting:
        output_root = Path(str(output_setting))
    else:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_root = Path("results") / "profile_matrix" / stamp
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    runs = build_runs(
        config,
        output_root=output_root,
        python_exe=args.python,
        benchmark_filter=set(args.benchmark or []),
        profile_filter=set(args.profile or []),
    )
    if not runs:
        raise SystemExit("matrix selection produced no runs")

    if not args.dry_run:
        # Keep the exact experiment input beside the generated results.
        target_config = output_root / "matrix.yaml"
        target_config.parent.mkdir(parents=True, exist_ok=True)
        target_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    records = run_matrix(
        runs,
        output_root=output_root,
        config_path=config_path,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
    )
    failures = [r for r in records if r["status"] == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

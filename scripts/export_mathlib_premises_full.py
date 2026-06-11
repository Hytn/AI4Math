#!/usr/bin/env python3
"""scripts/export_mathlib_premises_full.py — Mathlib 全量前提导出 (driver)

与旧脚本的关系: ``scripts/export_mathlib_premises.py`` 的正则文本抽取
保留不动 (无 Lean 环境时的 best-effort 路径)。本脚本是**新增**的
environment-based 全量导出驱动, 产出 ~10^5 量级的真实语料, 解决
"data/premises 只有 404 条、检索召回物理上限极低"的核心缺口。

流程:
  1. 把 scripts/lean/ExportPremises.lean 拷入目标 lake 项目
     (默认 data/miniF2F, 它已依赖 Mathlib 并 pin v4.24.0);
  2. `lake env lean ExportPremises.lean` → mathlib_premises_raw.jsonl;
  3. 后处理: 去重、长度截断、domain 标注 (按 module 前缀),
     写入 data/premises/mathlib_full.jsonl;
  4. local_tfidf provider 与 PremiseSearchTool 自动加载该文件,
     无需任何其他配置。

用法:
    python scripts/export_mathlib_premises_full.py \
        --lake-project data/miniF2F \
        -o data/premises/mathlib_full.jsonl

    # 已手动跑过 Lean 步骤时, 仅做后处理:
    python scripts/export_mathlib_premises_full.py \
        --raw data/miniF2F/mathlib_premises_raw.jsonl \
        -o data/premises/mathlib_full.jsonl

要求: 目标项目已 `lake exe cache get && lake build` (README Quickstart
第 2 步)。全量 pp 约需 10-30 分钟。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEAN_SCRIPT = os.path.join(HERE, "lean", "ExportPremises.lean")

# module 前缀 → domain 标签 (供 domain_filter 用)
_DOMAIN_MAP = [
    ("Mathlib.NumberTheory", "number_theory"),
    ("Mathlib.Algebra", "algebra"),
    ("Mathlib.GroupTheory", "algebra"),
    ("Mathlib.RingTheory", "algebra"),
    ("Mathlib.FieldTheory", "algebra"),
    ("Mathlib.LinearAlgebra", "algebra"),
    ("Mathlib.Analysis", "analysis"),
    ("Mathlib.MeasureTheory", "analysis"),
    ("Mathlib.Topology", "topology"),
    ("Mathlib.Combinatorics", "combinatorics"),
    ("Mathlib.Probability", "probability"),
    ("Mathlib.Order", "order"),
    ("Mathlib.Data.Nat", "nat"),
    ("Mathlib.Data.Int", "int"),
    ("Mathlib.Data.Real", "real"),
    ("Mathlib.Data", "data"),
    ("Mathlib.Logic", "logic"),
    ("Mathlib.SetTheory", "set_theory"),
    ("Mathlib.CategoryTheory", "category_theory"),
    ("Mathlib.Geometry", "geometry"),
]


def domain_of(module: str) -> str:
    for prefix, dom in _DOMAIN_MAP:
        if module.startswith(prefix):
            return dom
    return ""


def run_lean_export(lake_project: str) -> str:
    """在 lake 项目内执行 Lean 导出, 返回 raw jsonl 路径。"""
    if not os.path.isdir(lake_project):
        sys.exit(f"Error: lake project not found: {lake_project}")
    if not os.path.exists(os.path.join(lake_project, "lakefile.lean")) and \
       not os.path.exists(os.path.join(lake_project, "lakefile.toml")):
        sys.exit(f"Error: {lake_project} is not a lake project "
                 f"(no lakefile.lean/lakefile.toml)")
    if shutil.which("lake") is None:
        sys.exit("Error: `lake` not on PATH. "
                 "Run `source ~/.elan/env` first (README step 2).")

    dst = os.path.join(lake_project, "ExportPremises.lean")
    shutil.copyfile(LEAN_SCRIPT, dst)
    print(f"[1/2] Running Lean export in {lake_project} "
          f"(10-30 min on full Mathlib)...")
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", "ExportPremises.lean"],
            cwd=lake_project, capture_output=True, text=True)
    finally:
        # 不在他人项目里留垃圾
        try:
            os.remove(dst)
        except OSError:
            pass
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:])
        sys.exit(f"Error: Lean export failed (exit {proc.returncode}). "
                 f"Is the project built? Try `lake exe cache get && "
                 f"lake build` inside {lake_project}.")
    print(proc.stdout.strip()[-500:])
    raw = os.path.join(lake_project, "mathlib_premises_raw.jsonl")
    if not os.path.exists(raw):
        sys.exit("Error: Lean export produced no output file.")
    return raw


def postprocess(raw_path: str, out_path: str,
                max_statement_chars: int = 1200) -> dict:
    """去重 + domain 标注 + 截断超长陈述, 写最终 jsonl。"""
    print(f"[2/2] Post-processing {raw_path} ...")
    seen: set[str] = set()
    n_in = n_out = n_bad = n_dup = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(raw_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            name = e.get("name") or ""
            stmt = (e.get("statement") or "").strip()
            if not name or not stmt:
                n_bad += 1
                continue
            if name in seen:
                n_dup += 1
                continue
            seen.add(name)
            if len(stmt) > max_statement_chars:
                stmt = stmt[:max_statement_chars] + " …"
            module = e.get("module", "")
            fout.write(json.dumps({
                "name": name,
                "statement": stmt,
                "module": module,
                "domain": domain_of(module),
                "kind": e.get("kind", "theorem"),
            }, ensure_ascii=False) + "\n")
            n_out += 1
    stats = {"input": n_in, "written": n_out,
             "duplicates": n_dup, "malformed": n_bad}
    print(f"Done: {n_out} premises → {out_path}  "
          f"(dups={n_dup}, malformed={n_bad})")
    if n_out < 50_000:
        print("⚠ Fewer than 50k entries — full Mathlib should yield ~10^5 "
              "theorems. Check that the lake project actually imports "
              "all of Mathlib (data/miniF2F does).")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lake-project", default="data/miniF2F",
                    help="依赖 Mathlib 且已构建的 lake 项目 (默认 data/miniF2F)")
    ap.add_argument("--raw", default="",
                    help="已有的 raw jsonl (跳过 Lean 步骤, 仅后处理)")
    ap.add_argument("-o", "--output",
                    default="data/premises/mathlib_full.jsonl")
    ap.add_argument("--max-statement-chars", type=int, default=1200)
    args = ap.parse_args()

    raw = args.raw or run_lean_export(args.lake_project)
    postprocess(raw, args.output, args.max_statement_chars)


if __name__ == "__main__":
    main()

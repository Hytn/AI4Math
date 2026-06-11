#!/usr/bin/env python3
"""scripts/eval/fetch_matharena.py — 运行时拉取 MathArena 竞赛数据

⚠️ License: MathArena 数据集为 CC BY-NC-SA 4.0 —— **不要**把题目数据
vendor 进本仓库或随结果再分发; 本脚本只在用户本机运行时拉取到
data/MathArena/ (已在 .gitignore 习惯目录 data/ 下)。

数据源: https://huggingface.co/MathArena (每个竞赛一个 dataset repo,
字段: problem_idx / problem / answer / problem_type)。

依赖 (可选安装, 缺失时给出指引):
    pip install huggingface_hub datasets

用法:
    python scripts/eval/fetch_matharena.py --comp aime_2026
    python scripts/eval/fetch_matharena.py --comp hmmt_feb_2026 apex_2025

输出: data/MathArena/<comp>/problems.jsonl
    每行 {"problem_idx", "problem", "answer", "problem_type"}
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def fetch_one(comp: str, out_root: str) -> str:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit("需要: pip install datasets huggingface_hub\n"
                 "(MathArena 数据是 CC BY-NC-SA 4.0, 只做运行时拉取, "
                 "不随仓库分发。)")
    repo = f"MathArena/{comp}"
    print(f"Fetching {repo} ...")
    try:
        ds = load_dataset(repo, split="train")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"拉取 {repo} 失败: {e}\n"
                 f"可在 https://huggingface.co/MathArena 查看可用竞赛名。")
    out_dir = os.path.join(out_root, comp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "problems.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({
                "problem_idx": row.get("problem_idx", n),
                "problem": row.get("problem", ""),
                "answer": str(row.get("answer", "")),
                "problem_type": row.get("problem_type", []),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {n} problems → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comp", nargs="+", required=True,
                    help="竞赛名, e.g. aime_2026 hmmt_feb_2026 apex_2025")
    ap.add_argument("--out-root", default="data/MathArena")
    args = ap.parse_args()
    for c in args.comp:
        fetch_one(c, args.out_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/eval/eval_retrieval.py — 检索层自身的质量评测

为什么需要它: merge LeanSearch v2 / 换检索器之后, 如果只看端到端
pass@k, 检索层的改动无法归因 (prover 噪声远大于检索增益)。本脚本
直接在检索层算 Recall@K / nDCG@K / MRR, 与 LeanSearch v2 论文的
评测口径一致, 让"检索是否变好了"成为可独立验证的命题。

输入 qrels (jsonl), 每行:
    {"query": "commutativity of addition on naturals",
     "goal_state": "⊢ ∀ n m : ℕ, n + m = m + n",      # 可选
     "relevant": ["Nat.add_comm", "add_comm"]}

用法:
    python scripts/eval/eval_retrieval.py \
        --qrels data/retrieval_qrels/sample.jsonl \
        --providers leansearch_v2,local_tfidf \
        --k 1 5 10 \
        --cache results/retrieval_eval/cache.jsonl

    # 冻结快照重放 (复现已发布的数字):
    AI4MATH_RETRIEVAL_CACHE_MODE=ro python scripts/eval/eval_retrieval.py ...

输出: 终端表格 + --out 指定的 json。每个 provider 单独评 (不混合),
另附 providers 串联 (MultiRetriever) 的整体指标。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from prover.premise.providers import (  # noqa: E402
    MultiRetriever, build_providers)


def load_qrels(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"qrels line {ln}: bad json: {e}")
            if not rec.get("query") or not rec.get("relevant"):
                sys.exit(f"qrels line {ln}: needs 'query' and 'relevant'")
            out.append(rec)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2)
              for i, name in enumerate(ranked[:k]) if name in relevant)
    ideal = sum(1.0 / math.log2(i + 2)
                for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, name in enumerate(ranked):
        if name in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(searcher, qrels: list[dict], ks: list[int],
             label: str) -> dict:
    max_k = max(ks)
    agg = {f"recall@{k}": 0.0 for k in ks}
    agg.update({f"ndcg@{k}": 0.0 for k in ks})
    agg["mrr"] = 0.0
    n_degraded_queries = 0

    for rec in qrels:
        rel = set(rec["relevant"])
        if isinstance(searcher, MultiRetriever):
            hits, degraded = searcher.search(
                rec["query"], top_k=max_k,
                goal_state=rec.get("goal_state", ""))
            if degraded:
                n_degraded_queries += 1
        else:
            hits = searcher.search(rec["query"], top_k=max_k,
                                   goal_state=rec.get("goal_state", ""))
        ranked = [h.name for h in hits]
        for k in ks:
            agg[f"recall@{k}"] += recall_at_k(ranked, rel, k)
            agg[f"ndcg@{k}"] += ndcg_at_k(ranked, rel, k)
        agg["mrr"] += mrr(ranked, rel)

    n = max(1, len(qrels))
    result = {m: round(v / n, 4) for m, v in agg.items()}
    result["n_queries"] = len(qrels)
    result["degraded_queries"] = n_degraded_queries
    result["label"] = label
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--providers", required=True,
                    help="逗号分隔, e.g. leansearch_v2,loogle,local_tfidf")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--cache", default="",
                    help="检索快照路径 (默认读 AI4MATH_RETRIEVAL_CACHE)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    providers = build_providers(args.providers, cache_path=args.cache)
    if not providers:
        sys.exit("No providers constructed — check --providers spec.")

    rows = []
    # 每个 provider 单独评
    for p in providers:
        rows.append(evaluate(p, qrels, args.k, label=p.name))
    # 串联整体
    if len(providers) > 1:
        rows.append(evaluate(MultiRetriever(providers=providers),
                             qrels, args.k, label="(combined)"))

    cols = ["label"] + [f"recall@{k}" for k in args.k] + \
           [f"ndcg@{k}" for k in args.k] + ["mrr", "degraded_queries"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows))
              for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("─" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if any(r["degraded_queries"] for r in rows):
        print("\n⚠ degraded_queries > 0: 部分查询有 provider 不可用/失败。"
              "在线 provider 请检查网络或改用快照 ro 模式。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"qrels": args.qrels, "k": args.k,
                       "results": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

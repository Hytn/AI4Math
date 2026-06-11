#!/usr/bin/env python3
"""scripts/eval/matharena_informal.py — MathArena 非形式化基线评测

定位: 这是**informal reasoner 选型基准**, 不是形式化评测。Hilbert 论文
的核心实验结论之一是 reasoner 强弱比 prover 强弱更影响最终形式化成功
率 —— 在把一个通用 LLM 配进 hilbert/heterogeneous 等双角色 profile
之前, 先用本脚本在无污染的最新竞赛题上量它的 informal 水平。

口径对齐 MathArena 官方: 每题独立跑 n 次 (官方 n=4), 报 avg@n 与
每次成本估计; final-answer 判定做数值归一化 (去逗号/空格/$, 分数
化简比较)。proof-based 赛道 (USAMO/IMO) 需要人评或 judge, 不在本
脚本范围 — 用官方 eth-sri/matharena 的 judge 流程。

用法 (先 fetch_matharena.py 拉数据):
    python scripts/eval/matharena_informal.py \
        --comp aime_2026 \
        --provider anthropic --model claude-opus-4-5 \
        --n-runs 4 --out results/matharena/aime_2026_claude.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.brain.async_llm_provider import create_async_provider  # noqa: E402

_ANSWER_TAG = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_LINE = re.compile(
    r"(?:final answer|answer)\s*[:=]?\s*([^\n]+)", re.IGNORECASE)

SYSTEM = (
    "You are an expert competition mathematician. Solve the problem and "
    "put your final answer in \\boxed{}.")


def extract_answer(text: str) -> str:
    m = list(_ANSWER_TAG.finditer(text or ""))
    if m:
        return m[-1].group(1).strip()
    m2 = list(_FINAL_LINE.finditer(text or ""))
    if m2:
        return m2[-1].group(1).strip()
    return ""


def normalize(ans: str) -> str:
    a = (ans or "").strip()
    a = a.replace("$", "").replace(",", "").replace(" ", "")
    a = a.strip(".")
    # \frac{a}{b} / a/b → 既约分数; 整数去前导零
    m = re.fullmatch(r"\\?d?frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m:
        a = f"{m.group(1)}/{m.group(2)}"
    try:
        if re.fullmatch(r"-?\d+/-?\d+", a):
            return str(Fraction(a))
        if re.fullmatch(r"-?\d+", a):
            return str(int(a))
        if re.fullmatch(r"-?\d*\.\d+", a):
            return str(float(a))
    except (ValueError, ZeroDivisionError):
        pass
    return a.lower()


def is_correct(model_ans: str, gold: str) -> bool:
    return normalize(model_ans) != "" and normalize(model_ans) == normalize(gold)


async def run(args):
    comp_file = Path(args.data_root) / args.comp / "problems.jsonl"
    if not comp_file.exists():
        sys.exit(f"{comp_file} 不存在 — 先跑 "
                 f"python scripts/eval/fetch_matharena.py --comp {args.comp}")
    problems = [json.loads(l) for l in
                comp_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        problems = problems[:args.limit]

    provider = create_async_provider({
        "provider": args.provider, "model": args.model,
        "api_base": args.api_base, "api_key": args.api_key,
        "omit_temperature": args.omit_temperature,
    })

    sem = asyncio.Semaphore(args.concurrency)
    per_problem: dict = {}
    total_tokens = 0

    async def one_attempt(p, run_idx):
        nonlocal total_tokens
        async with sem:
            try:
                resp = await provider.chat(
                    system=SYSTEM,
                    messages=[{"role": "user", "content": p["problem"]}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens)
            except Exception as e:  # noqa: BLE001
                return {"correct": False, "answer": "",
                        "error": str(e)[:200]}
            content = getattr(resp, "content", "") or ""
            # LLMResponse 的成本字段是 tokens_in / tokens_out
            total_tokens += (int(getattr(resp, "tokens_in", 0) or 0)
                             + int(getattr(resp, "tokens_out", 0) or 0))
            ans = extract_answer(content)
            return {"correct": is_correct(ans, p["answer"]),
                    "answer": ans}

    for p in problems:
        attempts = await asyncio.gather(
            *[one_attempt(p, i) for i in range(args.n_runs)])
        per_problem[str(p["problem_idx"])] = {
            "gold": p["answer"],
            "attempts": attempts,
            "avg": sum(a["correct"] for a in attempts) / max(1, args.n_runs),
        }
        idx = p["problem_idx"]
        print(f"  #{idx}: avg@{args.n_runs} = "
              f"{per_problem[str(idx)]['avg']:.2f}")

    avg = (sum(v["avg"] for v in per_problem.values())
           / max(1, len(per_problem)))
    summary = {
        "comp": args.comp, "model": args.model, "provider": args.provider,
        "n_runs": args.n_runs, "n_problems": len(per_problem),
        "avg_score": round(avg, 4),
        "total_tokens": total_tokens,     # 成本对账维度 (见 README 评测原则)
        "per_problem": per_problem,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        Path(args.out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nWrote {args.out}")
    print(f"\n{args.comp}  {args.model}  avg@{args.n_runs} = {avg:.4f}  "
          f"({len(per_problem)} problems, ~{total_tokens} tokens)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comp", required=True)
    ap.add_argument("--data-root", default="data/MathArena")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--omit-temperature", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--n-runs", type=int, default=4,
                    help="官方口径 avg@4")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

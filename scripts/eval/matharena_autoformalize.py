#!/usr/bin/env python3
"""scripts/eval/matharena_autoformalize.py — MathArena 形式化线 (第 2 步)

流程定位 (双线中的 formal 线):
    fetch_matharena.py  →  本脚本  →  run_eval.py --benchmark matharena

本脚本把 final-answer 竞赛题转成 **answer-aware** 的 Lean 4 定理:
答案已知 (gold), 所以形式化目标是 "题设条件下, 所求量 = gold"
(miniF2F/PutnamBench solution-substituted 同款形态), 而不是
NL_EXISTENCE 桥的存在式形态 —— 存在式会让 prover 用平凡见证作弊。

输出: data/MathArena/<comp>/formalized.jsonl, 每行:
    {"problem_idx", "problem", "answer", "statement",
     "formalizer_model", "flagged": bool}
``statement`` 以 ``:= by sorry`` 结尾, 供 benchmarks/datasets/matharena
loader 直接装载。``flagged=true`` 表示形式化产物未通过本脚本的静态
红线 (无 sorry 替换点 / 含 native_decide 等), 默认仍写出但 loader
会跳过 —— 人工修订后把 flagged 改 false 即可纳入。

⚠️ 自动形式化本身是误差源: 评测报告里必须注明 formalizer 模型与
   flagged 比例 (faithfulness 未经人工核对的形式化结果不可与
   miniF2F 等人工基准直接比较)。

用法:
    python scripts/eval/matharena_autoformalize.py \
        --comp aime_2026 --provider anthropic --model claude-opus-4-5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.brain.async_llm_provider import create_async_provider  # noqa: E402

SYSTEM = """\
You are a Lean 4 (Mathlib) autoformalizer for competition problems whose
final numeric answer is already known. Translate the problem into ONE
Lean 4 theorem asserting that the requested quantity equals the given
answer. Output rules:

1. Output the `theorem` declaration ONLY — no Markdown fences, no
   commentary, no `import` lines.
2. Name it `matharena_q{IDX}`. End the declaration with `:= by sorry`.
3. Encode the problem's hypotheses as explicit binders/hypotheses; the
   conclusion must pin the answer, e.g. `... : f 2026 = 1234 := by sorry`.
   Never use an existential as the top-level conclusion.
4. Use Mathlib-standard types (ℕ ℤ ℚ ℝ, Finset, Nat.Prime, ...). For
   AIME-style integer answers prefer ℕ/ℤ. Encode fractions exactly
   (e.g. (7 : ℚ) / 3), never as decimals.
5. If part of the problem cannot be encoded faithfully, leave a
   `/- TODO: ... -/` comment inside the statement at the imprecise spot.
"""

_FENCE = re.compile(r"```(?:lean4?)?\s*\n?(.*?)```", re.DOTALL)
_RED_FLAGS = ("native_decide", "axiom ", "admit", "maxHeartbeats 0")


def clean_statement(raw: str, idx) -> tuple[str, bool]:
    """抽出 theorem 声明并做静态红线检查; 返回 (statement, flagged)。"""
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # 截到 theorem 开头
    t = text.find("theorem")
    if t > 0:
        text = text[t:]
    flagged = False
    if not text.startswith("theorem"):
        flagged = True
    if ":= by sorry" not in text.replace(" :=  by", " := by"):
        # 统一补尾 (有些模型给 `:= sorry`)
        if text.rstrip().endswith(":= sorry"):
            text = text.rstrip()[: -len(":= sorry")] + ":= by sorry"
        else:
            flagged = True
    if any(rf in text for rf in _RED_FLAGS):
        flagged = True
    if f"matharena_q{idx}" not in text:
        # 不致命, 但规范化名字便于追踪
        text = re.sub(r"theorem\s+[A-Za-z0-9_.']+",
                      f"theorem matharena_q{idx}", text, count=1)
    return text, flagged


async def run(args):
    comp_dir = Path(args.data_root) / args.comp
    src = comp_dir / "problems.jsonl"
    if not src.exists():
        sys.exit(f"{src} 不存在 — 先跑 fetch_matharena.py --comp {args.comp}")
    problems = [json.loads(l) for l in
                src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        problems = problems[:args.limit]

    provider = create_async_provider({
        "provider": args.provider, "model": args.model,
        "api_base": args.api_base, "api_key": args.api_key,
        "omit_temperature": args.omit_temperature,
    })
    sem = asyncio.Semaphore(args.concurrency)

    async def formalize(p):
        idx = p["problem_idx"]
        user = (f"Problem (index {idx}):\n{p['problem']}\n\n"
                f"Known final answer: {p['answer']}\n\n"
                f"Formalize as instructed.")
        async with sem:
            try:
                resp = await provider.generate(
                    system=SYSTEM.replace("{IDX}", str(idx)), user=user,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens)
            except Exception as e:  # noqa: BLE001
                return {**p, "statement": "", "flagged": True,
                        "error": str(e)[:200],
                        "formalizer_model": args.model}
        stmt, flagged = clean_statement(
            getattr(resp, "content", "") or "", idx)
        return {"problem_idx": idx, "problem": p["problem"],
                "answer": p["answer"], "statement": stmt,
                "flagged": flagged, "formalizer_model": args.model}

    rows = await asyncio.gather(*[formalize(p) for p in problems])
    out = comp_dir / "formalized.jsonl"
    n_flag = sum(r["flagged"] for r in rows)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} formalized → {out}  (flagged={n_flag})")
    if n_flag:
        print(f"⚠ {n_flag} 条被 flag — 人工修订 statement 并把 flagged 置 "
              f"false 后才会被 loader 装载。")
    print(f"下一步: python run_eval.py --benchmark matharena "
          f"--path {args.data_root} --split {args.comp} --profile repair ...")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comp", required=True)
    ap.add_argument("--data-root", default="data/MathArena")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="")
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--omit-temperature", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

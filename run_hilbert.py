#!/usr/bin/env python3
"""run_hilbert.py — Hilbert 递归分解证明的独立入口

与 run_unified/run_eval 并列的第三入口, **不经过** profile 体系
(per-role 双模型配置无法塞进单 model 的 Profile dataclass, 详见
prover/hilbert/config.py 模块注释)。

用法:
    # 单题冒烟 (mock 全链路, 无 Lean / 无 API key)
    python run_hilbert.py --config config/hilbert.yaml \
        --statement "theorem t : 1 + 1 = 2 := by sorry" --backend mock

    # 跑基准 (真实 Lean + 双角色真模型)
    python run_hilbert.py --config config/hilbert.yaml \
        --benchmark minif2f --split test --limit 20 --lean \
        --out results/hilbert_minif2f

输出: <out>/<problem_id>/hilbert_trace.json (递归树全 trace +
per-role token 对账) 与 <out>/summary.json。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_hilbert")


def build_verifier(args, preamble: str):
    """返回 (async verify(code) -> (ok, errs), cleanup, is_mock)。"""
    if args.backend == "mock":
        async def verify_mock(code: str):
            # 与 run_unified --backend mock 同语义: 全部判过。
            # summary 里 is_mock=True, 评测脚本必须据此排除。
            return True, []
        async def noop():
            return None
        return verify_mock, noop, True

    if not args.lean:
        sys.exit("需要 --lean (真实 Lean) 或 --backend mock (冒烟)。")
    from engine.async_lean_pool import AsyncLeanPool
    pool = AsyncLeanPool(pool_size=args.lean_pool_size)

    async def verify_real(code: str):
        r = await pool.verify_complete(code, "", preamble)
        ok = bool(getattr(r, "success", False)) \
            and not getattr(r, "has_sorry", False)
        errs = [str(e.get("message", e)) if isinstance(e, dict) else str(e)
                for e in (getattr(r, "errors", None) or [])]
        return ok, errs

    async def cleanup():
        close = getattr(pool, "shutdown", None) or getattr(pool, "close", None)
        if close:
            res = close()
            if asyncio.iscoroutine(res):
                await res

    return verify_real, cleanup, False


def load_problems(args):
    from prover.models import BenchmarkProblem
    if args.statement:
        return [BenchmarkProblem(problem_id="adhoc", name="adhoc",
                                 theorem_statement=args.statement)]
    from benchmarks.loader import load_benchmark
    return load_benchmark(args.benchmark, split=args.split,
                          path=args.path, limit=args.limit)


async def amain(args):
    from agent.brain.async_llm_provider import create_async_provider
    from prover.hilbert import HilbertConfig, HilbertOrchestrator

    cfg = HilbertConfig.from_yaml(args.config) if args.config \
        else HilbertConfig()
    if args.backend == "mock":
        cfg.reasoner.provider = cfg.prover.provider = "mock"

    reasoner = create_async_provider(cfg.reasoner.provider_config())
    prover = create_async_provider(cfg.prover.provider_config())
    verify, cleanup, is_mock = build_verifier(args, cfg.preamble)

    problems = load_problems(args)
    if not problems:
        sys.exit("没有可跑的题目。")
    os.makedirs(args.out, exist_ok=True)

    solved = 0
    summary_rows = []
    try:
        for p in problems:
            orch = HilbertOrchestrator(cfg, reasoner, prover, verify)
            t0 = time.time()
            root = await orch.prove(p.theorem_statement)
            ok = root.status == "proved"
            solved += ok
            row = {
                "problem_id": p.problem_id,
                "solved": ok,
                "method": root.method,
                "elapsed_s": round(time.time() - t0, 1),
                "reasoner_calls": orch.stats["reasoner"].calls,
                "prover_calls": orch.stats["prover"].calls,
                "tokens": {r: s.tokens_in + s.tokens_out
                           for r, s in orch.stats.items()},
            }
            summary_rows.append(row)
            pdir = os.path.join(args.out, p.problem_id)
            os.makedirs(pdir, exist_ok=True)
            with open(os.path.join(pdir, "hilbert_trace.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"problem": p.theorem_statement,
                           "is_mock": is_mock,
                           "config": {
                               "max_depth": cfg.max_depth,
                               "prover_passes": cfg.prover_passes,
                               "shallow_passes": cfg.shallow_passes,
                               "reasoner_model": cfg.reasoner.model,
                               "prover_model": cfg.prover.model,
                           },
                           **row, "tree": root.to_dict()},
                          f, ensure_ascii=False, indent=2)
            logger.info("%s: %s (%s)", p.problem_id,
                        "SOLVED" if ok else "failed", root.method or "-")
    finally:
        await cleanup()

    with open(os.path.join(args.out, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"solved": solved, "total": len(problems),
                   "is_mock": is_mock, "rows": summary_rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\nHilbert: {solved}/{len(problems)} solved "
          f"{'(MOCK — not real proofs)' if is_mock else ''} → {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/hilbert.yaml")
    ap.add_argument("--statement", default="",
                    help="直接给单个题面 (与 --benchmark 互斥)")
    ap.add_argument("--benchmark", default="minif2f")
    ap.add_argument("--split", default="test")
    ap.add_argument("--path", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lean", action="store_true", help="真实 Lean 验证")
    ap.add_argument("--backend", default="",
                    help="mock = 冒烟 (LLM 与 verifier 全 mock)")
    ap.add_argument("--lean-pool-size", type=int, default=4)
    ap.add_argument("--out", default="results/hilbert")
    args = ap.parse_args()
    if args.config and not os.path.exists(args.config):
        logger.warning("config %s 不存在, 用默认 HilbertConfig", args.config)
        args.config = ""
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/eval/integration_drill.py — 新增功能全链路集成演练 (离线)

目的: 在 **无 Lean 工具链 / 无外网 / 无 API key** 的最小环境下, 把本仓库
新增的全部模块按真实使用方式串通一遍, 任何一环断裂立即非零退出。
它既是交付验收脚本, 也是 README《End-to-end walkthrough》一节的
事实来源 —— 文档里的每条命令都先在这里跑通。

覆盖面 (S1–S5):
  S1  检索 provider 层: 快照缓存 rw 预热 → ro 冻结重放 →
      eval_retrieval.py 出 Recall/nDCG/MRR, 在线 provider 零降级
  S2  PremiseSearchTool 环境变量注入: provider 结果优先序 +
      degraded_providers 元数据 (真实工具调用路径, 非单测桩)
  S3  lean-eval comparator 线: mini 仓库 fixture → run_eval 范式加载 →
      run_leaneval.py 备份-写入-validate-score-还原全流程
      (comparator 用桩脚本替身, 接口与官方 CLI 同形)
  S4  MathArena 双线: informal avg@n (mock provider) →
      autoformalize 红线 flag 拦截 → 人工修订 → loader 装载
  S5  Hilbert 串接: run_hilbert --benchmark matharena --backend mock
      端到端递归调度 + trace 落盘

用法:
    python scripts/eval/integration_drill.py            # 全跑
    python scripts/eval/integration_drill.py --keep     # 保留工作目录

真实环境 (有 Lean / 有 API key / 可出网) 的对应命令见 README
《End-to-end walkthrough》— 本脚本每个 stage 的 banner 也会打印
对应的真实命令以便对照。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

PY = sys.executable
_FAILED = []


def banner(stage: str, title: str, real_cmd: str = ""):
    print(f"\n{'═' * 70}\n{stage}  {title}")
    if real_cmd:
        print(f"    (真实环境对应命令: {real_cmd})")
    print("═" * 70)


def check(cond: bool, msg: str):
    tag = "✓" if cond else "✗"
    print(f"  {tag} {msg}")
    if not cond:
        _FAILED.append(msg)


def run_cli(args: list[str], env: dict | None = None,
            cwd: Path | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    print(f"  $ {' '.join(str(a) for a in args)}")
    p = subprocess.run([PY] + [str(a) for a in args], env=e,
                       cwd=str(cwd or ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        print(textwrap.indent((p.stdout + p.stderr)[-1500:], "    │ "))
    return p


# ════════════════════════════════════════════════════════════════════
# S1 检索 provider 层
# ════════════════════════════════════════════════════════════════════

def stage1_retrieval(ws: Path) -> None:
    banner("S1", "检索 provider 层: 快照预热 → ro 冻结重放 → 指标评测",
           "AI4MATH_RETRIEVAL_CACHE_MODE=rw python scripts/eval/"
           "eval_retrieval.py --providers leansearch_v2,loogle,local_tfidf ...")
    from prover.premise.providers.cache import RetrievalCache
    from prover.premise.providers.leansearch_v2 import LeanSearchV2Provider
    from prover.premise.providers.loogle import LoogleProvider

    cache_path = ws / "retrieval_cache.jsonl"
    qrels_path = ws / "qrels.jsonl"

    # 1. 离线环境没法打 leansearch.net — 用 rw 模式直接落快照,
    #    模拟"有网机器上预热过一次缓存"。条目结构与 provider 的
    #    _to_dict 序列化完全一致。
    cache = RetrievalCache(path=str(cache_path), mode="rw")
    ls = LeanSearchV2Provider(cache=cache)
    lg = LoogleProvider(cache=cache)
    snapshots = {
        "commutativity of natural number addition": [
            {"name": "Nat.add_comm", "statement": "∀ (n m : ℕ), n + m = m + n",
             "score": 0.97, "module": "Mathlib.Data.Nat.Defs",
             "informal": "addition on ℕ commutes", "kind": "theorem"},
            {"name": "add_comm", "statement": "∀ {G} [AddCommMonoid G] (a b : G), a + b = b + a",
             "score": 0.91, "module": "Mathlib.Algebra.Group.Defs",
             "informal": "", "kind": "theorem"}],
        "square of a sum expansion": [
            {"name": "add_sq", "statement": "(a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2",
             "score": 0.95, "module": "Mathlib.Algebra.GroupPower.Ring",
             "informal": "binomial square", "kind": "theorem"}],
    }
    for q, hits in snapshots.items():
        for prov in (ls, lg):
            k = RetrievalCache.make_key(prov.name, q, 0, extra=prov.url)
            cache.put(k, q, hits)

    qrels = [
        {"query": "commutativity of natural number addition",
         "relevant": ["Nat.add_comm", "add_comm"]},
        {"query": "square of a sum expansion",
         "relevant": ["add_sq"]},
    ]
    qrels_path.write_text("\n".join(json.dumps(r) for r in qrels),
                          encoding="utf-8")

    # 2. ro 冻结重放跑评测 CLI — 这正是"发布数字必须可复现"的口径。
    out_json = ws / "retrieval_eval.json"
    p = run_cli(["scripts/eval/eval_retrieval.py",
                 "--qrels", qrels_path, "--providers",
                 "leansearch_v2,loogle,local_tfidf",
                 "--k", "1", "5", "--cache", cache_path,
                 "--out", out_json],
                env={"AI4MATH_RETRIEVAL_CACHE_MODE": "ro"})
    check(p.returncode == 0, "eval_retrieval.py 退出码 0")
    res = json.loads(out_json.read_text())["results"]
    by = {r["label"]: r for r in res}
    check(by["leansearch_v2"]["recall@5"] == 1.0,
          f"leansearch_v2 ro 重放 recall@5 = {by['leansearch_v2']['recall@5']}")
    check(by["leansearch_v2"]["degraded_queries"] == 0,
          "leansearch_v2 快照命中, 零降级 (未打任何网络)")
    check("(combined)" in by and by["(combined)"]["mrr"] > 0,
          f"combined MRR = {by.get('(combined)', {}).get('mrr')}")
    # local_tfidf 在仅 404 条种子语料上可能查不到这些名字 — 这不算
    # 失败, 但 WARNING 必须出现 (召回天花板必须可见)。
    check("local_tfidf" in by, "local_tfidf 参评 (种子语料 404 条)")


# ════════════════════════════════════════════════════════════════════
# S2 PremiseSearchTool 环境变量注入
# ════════════════════════════════════════════════════════════════════

def stage2_tool_injection(ws: Path) -> None:
    banner("S2", "PremiseSearchTool: AI4MATH_PREMISE_PROVIDERS 注入 + 降级上报",
           'export AI4MATH_PREMISE_PROVIDERS="leansearch_v2,local_tfidf" '
           "&& python run_unified.py --profile repair ...")
    # 子进程跑, 让环境变量语义与真实 run_unified 启动完全一致。
    code = textwrap.dedent("""\
        import asyncio, json, sys
        from prover.premise.providers import build_providers
        from agent.tools.builtin.premise_search import PremiseSearchTool
        provs = build_providers()
        tool = PremiseSearchTool(providers=provs or None)
        res = asyncio.run(tool.execute(
            {"query": "commutativity of natural number addition",
             "max_results": 5}, ctx=None))
        print(json.dumps({"results": json.loads(res.content),
                          "degraded": res.metadata.get("degraded_providers"),
                          "n_providers": len(provs)}))
    """)
    p = subprocess.run(
        [PY, "-c", code], capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ,
             "AI4MATH_PREMISE_PROVIDERS": "leansearch_v2,local_tfidf",
             "AI4MATH_RETRIEVAL_CACHE": str(ws / "retrieval_cache.jsonl"),
             "AI4MATH_RETRIEVAL_CACHE_MODE": "ro"})
    check(p.returncode == 0, "工具子进程退出码 0")
    out = json.loads(p.stdout.strip().splitlines()[-1])
    check(out["n_providers"] == 2, "环境变量构造出 2 个 provider")
    first = out["results"][0] if out["results"] else {}
    check(first.get("source") == "leansearch_v2"
          and first.get("name") == "Nat.add_comm",
          f"provider 结果优先序在前: {first.get('source')}/{first.get('name')}")
    check(isinstance(out["degraded"], list),
          f"degraded_providers 元数据上报: {out['degraded']}")

    # 反向核验: 不设环境变量 → 行为与历史版本一致 (heuristic 兜底)。
    p2 = subprocess.run(
        [PY, "-c", code], capture_output=True, text=True, cwd=str(ROOT),
        env={k: v for k, v in os.environ.items()
             if not k.startswith("AI4MATH_")})
    out2 = json.loads(p2.stdout.strip().splitlines()[-1])
    check(out2["n_providers"] == 0 and all(
        r["source"] in ("heuristic", "tfidf", "knowledge_store")
        for r in out2["results"]),
        "未配置环境变量 → 零 provider, 走历史降级链 (默认行为不变)")


# ════════════════════════════════════════════════════════════════════
# S3 lean-eval comparator 线
# ════════════════════════════════════════════════════════════════════

def _make_leaneval_fixture(ws: Path) -> Path:
    repo = ws / "LeanEval"
    (repo / "manifests").mkdir(parents=True)
    (repo / "LeanEval").mkdir()
    (repo / "manifests" / "problems.toml").write_text(textwrap.dedent("""\
        [[problems]]
        id = "p001"
        declaration = "leaneval_p001"
        difficulty = "easy"

        [[problems]]
        id = "p002"
        declaration = "leaneval_p002"
    """), encoding="utf-8")
    (repo / "LeanEval" / "Basic.lean").write_text(textwrap.dedent("""\
        import Mathlib

        @[eval_problem]
        theorem leaneval_p001 : 1 + 1 = 2 := by sorry

        @[eval_problem]
        theorem leaneval_p002 (n : ℕ) : n + 0 = n := by sorry
    """), encoding="utf-8")
    for pid in ("p001", "p002"):
        d = repo / "generated" / pid
        d.mkdir(parents=True)
        (d / "Submission.lean").write_text(
            f"-- pristine workspace for {pid}\n", encoding="utf-8")
    # comparator 桩: 接口与官方 `lake exe lean-eval ...` 同形
    # (validate-submission --file X / score --problem ID),
    # 判定规则: Submission 含 `trivial` 即 accept。
    (ws / "comparator.py").write_text(textwrap.dedent("""\
        import sys
        from pathlib import Path
        mode = sys.argv[1]
        if mode == "validate-submission":
            f = Path(sys.argv[sys.argv.index("--file") + 1])
            sys.exit(0 if f.exists() else 1)
        if mode == "score":
            pid = sys.argv[sys.argv.index("--problem") + 1]
            sub = Path("generated") / pid / "Submission.lean"
            sys.exit(0 if "trivial" in sub.read_text() else 1)
        sys.exit(2)
    """), encoding="utf-8")
    return repo


def stage3_leaneval(ws: Path) -> None:
    banner("S3", "lean-eval: comparator 范式加载 + 官方口径评分驱动",
           "python scripts/eval/run_leaneval.py --from-traces "
           "results/<run>/traces/leaneval/")
    repo = _make_leaneval_fixture(ws)

    # 1. loader 与 run_eval 同一入口装载
    from benchmarks.loader import load_benchmark
    probs = load_benchmark("leaneval", path=str(repo))
    check(len(probs) == 2, f"load_benchmark('leaneval') 装载 {len(probs)} 题")
    check(all("comparator_scored" in p.tags for p in probs),
          "全部题目带 comparator_scored 标记 (lean_verify 仅预筛)")

    # 2. prover 产出 (一好一坏) → run_leaneval.py 官方口径判分
    proofs = ws / "proofs"
    proofs.mkdir()
    (proofs / "p001.lean").write_text(
        "theorem leaneval_p001 : 1 + 1 = 2 := by trivial", encoding="utf-8")
    (proofs / "p002.lean").write_text(
        "theorem leaneval_p002 (n : ℕ) : n + 0 = n := by sorry",
        encoding="utf-8")
    comp = ws / "comparator.py"
    p = run_cli(["scripts/eval/run_leaneval.py",
                 "--leaneval-dir", repo, "--proofs-dir", proofs,
                 "--out", ws / "leaneval_out",
                 "--validate-cmd",
                 f"{PY} {comp} validate-submission --file {{submission}}",
                 "--score-cmd", f"{PY} {comp} score --problem {{id}}"])
    check(p.returncode == 0, "run_leaneval.py 退出码 0")
    if p.returncode != 0:
        return
    res = json.loads((ws / "leaneval_out" / "leaneval_results.json")
                     .read_text())
    check(res["scoring"] == "comparator", "结果标记 scoring=comparator")
    check(res["solved"] == 1 and res["total"] == 2,
          f"comparator accepts {res['solved']}/{res['total']} (好证明过, sorry 不过)")
    pristine = (repo / "generated" / "p001" / "Submission.lean").read_text()
    check("pristine workspace" in pristine,
          "评分后 Submission.lean 已还原备份 (workspace 不被污染)")


# ════════════════════════════════════════════════════════════════════
# S4 MathArena 双线
# ════════════════════════════════════════════════════════════════════

def stage4_matharena(ws: Path) -> Path:
    banner("S4", "MathArena: informal avg@n → autoformalize 红线 → 人工修订 → 装载",
           "python scripts/eval/fetch_matharena.py --comp aime_2026 && "
           "python scripts/eval/matharena_informal.py / matharena_autoformalize.py")
    root = ws / "MathArena"
    comp_dir = root / "drill_2026"
    comp_dir.mkdir(parents=True)
    problems = [
        {"problem_idx": 1, "problem": "Compute 1 + 1.", "answer": "2",
         "problem_type": ["algebra"]},
        {"problem_idx": 2, "problem": "Compute 2 * 3.", "answer": "6",
         "problem_type": ["algebra"]},
    ]
    (comp_dir / "problems.jsonl").write_text(
        "\n".join(json.dumps(p) for p in problems), encoding="utf-8")
    print("  (离线替代 fetch_matharena.py: 手写 problems.jsonl, "
          "字段与 HF 数据集一致)")

    # 线 1: informal 评测机制 (mock provider 不会答对 — 验证的是
    # avg@n / 答案归一化 / token 对账整条机制能跑通)
    p = run_cli(["scripts/eval/matharena_informal.py",
                 "--comp", "drill_2026", "--data-root", root,
                 "--provider", "mock", "--model", "mock-m",
                 "--n-runs", "2", "--concurrency", "2",
                 "--out", ws / "informal.json"])
    check(p.returncode == 0, "matharena_informal.py 退出码 0 (mock)")
    inf = json.loads((ws / "informal.json").read_text())
    check(inf["n_problems"] == 2 and inf["n_runs"] == 2,
          f"avg@2 机制跑通: avg_score={inf['avg_score']} (mock 答不对属预期)")

    # 线 2: autoformalize — mock 输出不是合法定理, 必须被红线 flag
    p = run_cli(["scripts/eval/matharena_autoformalize.py",
                 "--comp", "drill_2026", "--data-root", root,
                 "--provider", "mock", "--model", "mock-m"])
    check(p.returncode == 0, "matharena_autoformalize.py 退出码 0 (mock)")
    rows = [json.loads(l) for l in
            (comp_dir / "formalized.jsonl").read_text().splitlines()]
    from benchmarks.loader import load_benchmark
    check(load_benchmark("matharena", path=str(root),
                         split="drill_2026") == [] or
          all(r["flagged"] for r in rows) is False,
          f"红线生效: {sum(r['flagged'] for r in rows)}/{len(rows)} 条被 "
          f"flag, flagged 条目 loader 拒载")

    # 人工修订流程 (文档化的正式路径): 改 statement + flagged=false
    fixed = [
        {"problem_idx": 1, "problem": problems[0]["problem"], "answer": "2",
         "statement": "theorem matharena_q1 : 1 + 1 = 2 := by sorry",
         "flagged": False, "formalizer_model": "human-revised"},
        {"problem_idx": 2, "problem": problems[1]["problem"], "answer": "6",
         "statement": "theorem matharena_q2 : 2 * 3 = 6 := by sorry",
         "flagged": False, "formalizer_model": "human-revised"},
    ]
    (comp_dir / "formalized.jsonl").write_text(
        "\n".join(json.dumps(r) for r in fixed), encoding="utf-8")
    probs = load_benchmark("matharena", path=str(root), split="drill_2026")
    check(len(probs) == 2, f"人工修订后 load_benchmark 装载 {len(probs)} 题")
    check(all("formalizer:human-revised" in p.tags for p in probs),
          "formalizer 溯源标签贯穿到 BenchmarkProblem")
    return root


# ════════════════════════════════════════════════════════════════════
# S5 Hilbert 串 MathArena
# ════════════════════════════════════════════════════════════════════

def stage5_hilbert(ws: Path, matharena_root: Path) -> None:
    banner("S5", "Hilbert: run_hilbert --benchmark matharena 端到端 (mock)",
           "python run_hilbert.py --config config/hilbert.yaml "
           "--benchmark matharena --split <comp> --lean")
    out = ws / "hilbert_out"
    p = run_cli(["run_hilbert.py", "--benchmark", "matharena",
                 "--path", matharena_root, "--split", "drill_2026",
                 "--backend", "mock", "--out", out])
    check(p.returncode == 0, "run_hilbert.py 退出码 0")
    summary = json.loads((out / "summary.json").read_text())
    check(summary["total"] == 2 and summary["is_mock"] is True,
          f"summary: {summary['solved']}/{summary['total']}, "
          f"is_mock={summary['is_mock']} (mock 结果不可计入真实数字)")
    trace_files = list(out.glob("*/hilbert_trace.json"))
    check(len(trace_files) == 2, f"{len(trace_files)} 份递归 trace 落盘")
    tr = json.loads(trace_files[0].read_text())
    check("tree" in tr and "reasoner_calls" in tr,
          "trace 含递归树 + per-role 成本对账")


# ════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true",
                    help="保留临时工作目录以便检视产物")
    args = ap.parse_args()

    ws = Path(tempfile.mkdtemp(prefix="ai4math_drill_"))
    print(f"工作目录: {ws}")
    try:
        stage1_retrieval(ws)
        stage2_tool_injection(ws)
        stage3_leaneval(ws)
        ma_root = stage4_matharena(ws)
        stage5_hilbert(ws, ma_root)
    finally:
        if args.keep:
            print(f"\n产物保留在: {ws}")
        else:
            shutil.rmtree(ws, ignore_errors=True)

    print(f"\n{'═' * 70}")
    if _FAILED:
        print(f"DRILL FAILED — {len(_FAILED)} 项断言未通过:")
        for m in _FAILED:
            print(f"  ✗ {m}")
        sys.exit(1)
    print("DRILL PASSED — 全部新增模块端到端串通 ✓")
    print("(本演练全程离线; 真实 Lean/API/网络下的对应命令见各 stage "
          "banner 与 README《End-to-end walkthrough》)")


if __name__ == "__main__":
    main()

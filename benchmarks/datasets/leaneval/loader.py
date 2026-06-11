"""benchmarks/datasets/leaneval/loader.py — leanprover/lean-eval 加载器

⚠️ 范式提示 (与其它 benchmark 的关键差异):

lean-eval (https://github.com/leanprover/lean-eval) 是 **comparator-based**
基准: 题目作者用 ``@[eval_problem]`` 在 ``LeanEval/`` 共享模块里标注可信
题面 (元数据在 ``manifests/problems.toml``), 工具为每题生成
``generated/<problem-id>/`` 的 comparator workspace。**一道题算解出当且
仅当 comparator 接受提交** —— 提交形态是修改 workspace 内的
``Submission.lean`` (及 ``Submission/`` 下的 .lean 文件), 而非"返回一个
证明字符串"。

因此:
  - 本 loader 负责把 lean-eval 仓库解析成 BenchmarkProblem 列表
    (供 prover 生成证明) —— 这部分与其它 loader 同构;
  - **评分不能复用本框架的 lean_verify (编译 + 无 sorry) 判定**;
    必须走 comparator: 见 ``scripts/eval/run_leaneval.py``, 它把
    prover 产出写回 workspace 并调 ``lake exe lean-eval
    validate-submission`` + comparator 评分。
  - ``run_eval.py --benchmark leaneval --lean-mode skip`` 可以用来批量
    "生成"证明 (不判定), 再交给 run_leaneval.py 终审。直接用
    ``--lean-mode real`` 得到的 solved 数字只是预筛, **不是**
    lean-eval 官方口径, eval 元数据中会写入
    ``scoring="prefilter_only"`` 提示这一点。

数据获取:
    git clone https://github.com/leanprover/lean-eval data/LeanEval

仓库结构 (loader 依赖的部分):
    LeanEval/                  可信题面模块 (.lean, 含 @[eval_problem])
    manifests/problems.toml    题目元数据 (id → module / 声明名)
    generated/<id>/            comparator workspace (生成后)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from prover.models import BenchmarkProblem

logger = logging.getLogger(__name__)

_EVAL_PROBLEM_RE = re.compile(
    r"@\[eval_problem[^\]]*\]\s*"
    r"(?:theorem|lemma)\s+([A-Za-z0-9_.'₀-₉]+)"
    r"(.*?)(?::=\s*by\s+sorry|:=\s*sorry)",
    re.DOTALL)


def _load_toml(path: Path) -> dict:
    """tomllib (py3.11+) → toml (pip) → 失败给出可操作的报错。"""
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        pass
    try:
        import toml  # type: ignore
        return toml.load(path)
    except ImportError:
        raise RuntimeError(
            "Parsing manifests/problems.toml needs Python 3.11+ (tomllib) "
            "or `pip install toml` on 3.10.")


def _parse_problem_modules(repo: Path) -> dict[str, dict]:
    """扫 LeanEval/ 下的 .lean, 抽出 @[eval_problem] 声明。

    返回 {declaration_name: {"statement": ..., "module_file": ...}}。
    statement 是从 attribute 到 `:= by sorry` 之间的完整声明头
    (即题面; 提交者要做的就是替换 sorry)。
    """
    out: dict[str, dict] = {}
    src_dir = repo / "LeanEval"
    if not src_dir.is_dir():
        return out
    for lf in sorted(src_dir.rglob("*.lean")):
        try:
            text = lf.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("leaneval: cannot read %s: %s", lf, e)
            continue
        for m in _EVAL_PROBLEM_RE.finditer(text):
            decl_name, header = m.group(1), m.group(2)
            stmt = f"theorem {decl_name}{header}".strip()
            # 统一成框架其它 loader 的题面形态: 以 `:= by sorry` 结尾的待证声明
            out[decl_name] = {
                "statement": stmt + " := by sorry",
                "module_file": str(lf.relative_to(repo)),
            }
    return out


def load(path: str = "data/LeanEval", split: str = "test",
         limit: int = 0) -> list[BenchmarkProblem]:
    """加载 lean-eval 题目。split 暂无官方切分, 参数保留接口一致性。"""
    repo = Path(path)
    manifest_path = repo / "manifests" / "problems.toml"
    if not manifest_path.exists():
        logger.error(
            "lean-eval: %s 不存在。请先:\n"
            "  git clone https://github.com/leanprover/lean-eval %s",
            manifest_path, path)
        return []

    try:
        manifest = _load_toml(manifest_path)
    except RuntimeError as e:
        logger.error("lean-eval: %s", e)
        return []

    decls = _parse_problem_modules(repo)

    # manifest 结构: 容错地接受 {problems: [ {...} ]} / {[<id>]: {...}} 两种
    entries: list[dict] = []
    if isinstance(manifest.get("problems"), list):
        entries = [e for e in manifest["problems"] if isinstance(e, dict)]
    else:
        for pid, meta in manifest.items():
            if isinstance(meta, dict):
                meta = dict(meta)
                meta.setdefault("id", pid)
                entries.append(meta)

    problems: list[BenchmarkProblem] = []
    n_missing_stmt = 0
    for e in entries:
        pid = str(e.get("id") or e.get("name") or "").strip()
        if not pid:
            continue
        decl = str(e.get("declaration") or e.get("decl")
                   or e.get("theorem") or pid)
        info = decls.get(decl) or decls.get(pid)
        if info is None:
            # workspace 已生成时可从 generated/<id>/Solution.lean 兜底取题面
            sol = repo / "generated" / pid / "Solution.lean"
            if sol.exists():
                info = {"statement": sol.read_text(encoding="utf-8"),
                        "module_file": str(sol.relative_to(repo))}
            else:
                n_missing_stmt += 1
                continue
        problems.append(BenchmarkProblem(
            problem_id=f"leaneval_{pid}",
            name=pid,
            theorem_statement=info["statement"],
            difficulty=str(e.get("difficulty", "unknown")),
            source="lean-eval",
            natural_language=str(e.get("description", "")),
            tags=["leaneval", "comparator_scored"]
                 + [str(t) for t in (e.get("tags") or [])],
            lean_preamble="import Mathlib\n",
        ))
        if limit and len(problems) >= limit:
            break

    if n_missing_stmt:
        logger.warning(
            "lean-eval: %d manifest entries had no parsable statement "
            "(neither @[eval_problem] in LeanEval/ nor a generated "
            "workspace). Run the repo's generation step first.",
            n_missing_stmt)
    logger.info("lean-eval: loaded %d problems from %s",
                len(problems), path)
    return problems

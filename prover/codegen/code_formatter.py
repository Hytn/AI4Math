"""prover/codegen/code_formatter.py — Lean 证明文本整形纯函数 (重建)

历史: ``prover.codegen`` 包在上传版本中并不存在, 但 ``run_mcts_eval.py``
(line 35) 仍 import 本模块的 ``extract_proof_body`` —— 与
``agent.brain.claude_provider`` 同批的 import 即崩既有缺陷。此处以
共享纯函数模块重建; ``prover.hilbert`` 的递归调度器同样引用本实现
(单一事实源, 行为由 tests/test_hilbert.py::TestPureHelpers 钉死)。

职责: 从模型输出/已抽取的 Lean 片段中归一出 ``by ...`` 证明体 —
容错 Markdown fence、带声明头的完整 theorem、裸 term 证明三种形态。
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:lean4?)?\s*\n?(.*?)```", re.DOTALL)
_SORRY_TAIL = re.compile(r":=\s*(?:by\s+)?sorry\s*$")


def statement_head(stmt: str) -> str:
    """去掉题面末尾的 ``:= by sorry`` / ``:= sorry``, 得到纯声明头。"""
    return _SORRY_TAIL.sub("", (stmt or "").strip()).strip()


def extract_proof_body(raw: str) -> str:
    """从模型输出抽 ``by ...`` 证明体。

    容错路径:
      - Markdown fence 包裹 → 取 fence 内;
      - 完整 ``theorem/lemma/example ... := <proof>`` → 取顶层 ``:=`` 后;
      - 裸 term 证明 (短单行) → 包成 ``by exact (term)`` 以统一组装。
    """
    text = (raw or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    if text.startswith(("theorem", "lemma", "example")):
        i = text.find(":=")
        if i >= 0:
            text = text[i + 2:].strip()
    if text.startswith(":="):
        text = text[2:].strip()
    if not text.startswith("by") and text:
        if "\n" not in text and len(text) < 400:
            text = f"by exact ({text})"
    return text.strip()

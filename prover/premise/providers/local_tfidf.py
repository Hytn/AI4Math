"""prover/premise/providers/local_tfidf.py — 本地 TF-IDF provider

把既有的 ``knowledge.tfidf_retriever.KnowledgeTFIDFRetriever`` +
``data/premises/*.jsonl`` 语料包装成 :class:`RetrieverProvider`。

不修改任何既有模块 —— jsonl 加载逻辑与
``agent/tools/builtin/premise_search.py::_load_premise_lemmas`` 保持
同一数据源约定 (data/premises/*.jsonl, 字段 name/statement/proof/domain),
但独立实现, 避免反向依赖 tool 层。

跑过 ``scripts/export_mathlib_premises_full.py`` 之后, 本 provider
自动获得全量 Mathlib 语料 (10^5 量级), 无需任何配置改动。
"""
from __future__ import annotations

import glob
import json
import logging
import os

from prover.premise.providers.base import (
    RetrievedPremise, RetrieverProvider, register_provider)

logger = logging.getLogger(__name__)


def load_premise_jsonl(dirs: list[str] | None = None) -> list[dict]:
    """加载 data/premises/*.jsonl 全部条目 (按 name 去重)。"""
    candidates = list(dirs) if dirs else []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(
        os.path.join(here, "..", "..", "..", "data", "premises")))
    candidates.append("data/premises")

    seen_files: set[str] = set()
    seen_names: set[str] = set()
    out: list[dict] = []
    for cand in candidates:
        if not cand or not os.path.isdir(cand):
            continue
        for fp in sorted(glob.glob(os.path.join(cand, "*.jsonl"))):
            ap = os.path.realpath(fp)
            if ap in seen_files:
                continue
            seen_files.add(ap)
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        name = entry.get("name") or ""
                        stmt = entry.get("statement") or ""
                        if not name or not stmt or name in seen_names:
                            continue
                        seen_names.add(name)
                        out.append({
                            "name": name,
                            "statement": stmt,
                            "proof": entry.get("proof", ""),
                            "domain": entry.get("domain", ""),
                            "module": entry.get("module", ""),
                        })
            except OSError as e:
                logger.warning("local_tfidf: failed to load %s: %s", fp, e)
    return out


@register_provider
class LocalTfidfProvider(RetrieverProvider):
    name = "local_tfidf"

    def __init__(self, premise_dirs: list[str] | None = None, **_):
        self._dirs = premise_dirs
        self._retriever = None
        self._init_failed = False
        self._corpus_size = 0

    def _ensure_index(self):
        if self._retriever is not None or self._init_failed:
            return
        try:
            from knowledge.tfidf_retriever import KnowledgeTFIDFRetriever
            lemmas = load_premise_jsonl(self._dirs)
            if not lemmas:
                self._init_failed = True
                logger.warning(
                    "local_tfidf: no premises found under data/premises/ — "
                    "provider degraded. Run "
                    "scripts/export_mathlib_premises_full.py to populate.")
                return
            r = KnowledgeTFIDFRetriever()
            r.index_lemmas(lemmas)
            self._retriever = r
            self._corpus_size = len(lemmas)
            if self._corpus_size < 5000:
                logger.warning(
                    "local_tfidf: corpus has only %d entries (Mathlib4 is "
                    "~10^5). Recall is physically capped — run "
                    "scripts/export_mathlib_premises_full.py.",
                    self._corpus_size)
        except Exception as e:  # noqa: BLE001
            self._init_failed = True
            logger.warning("local_tfidf init failed: %s", e)

    def available(self) -> bool:
        self._ensure_index()
        return self._retriever is not None

    def search(self, query: str, top_k: int = 10, *,
               goal_state: str = "") -> list[RetrievedPremise]:
        self._ensure_index()
        if self._retriever is None:
            return []
        try:
            hits = self._retriever.search(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            logger.warning("local_tfidf search failed: %s", e)
            return []
        return [
            RetrievedPremise(
                name=getattr(h, "name", ""),
                statement=getattr(h, "statement", ""),
                score=float(getattr(h, "score", 0.0)),
                source=self.name,
            )
            for h in hits if getattr(h, "name", "")
        ]

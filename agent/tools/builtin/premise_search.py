"""agent/tools/builtin/premise_search.py — Search Mathlib for relevant lemmas"""
from __future__ import annotations

import json
import logging

from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission

logger = logging.getLogger(__name__)

class PremiseSearchTool(Tool):
    name = "premise_search"
    description = (
        "Search Mathlib and the knowledge base for lemmas relevant to the "
        "current proof goal. Returns ranked results with type signatures."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language or Lean4 pattern to search for, "
                    "e.g. 'commutativity of addition' or 'Nat.add_comm'"
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 10)",
            },
            "domain_filter": {
                "type": "string",
                "description": (
                    "Optional domain filter: number_theory, algebra, "
                    "analysis, topology, combinatorics"
                ),
            },
            "goal_state": {
                "type": "string",
                "description": (
                    "Optional: the current Lean 4 proof state (as printed "
                    "by the REPL). When provided, state-conditioned "
                    "retrievers (if configured) use it for higher-precision "
                    "premise search."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self, knowledge_store=None, premise_db_path: str = "",
                 providers=None):
        """``providers`` (可选): list[RetrieverProvider] —
        来自 ``prover.premise.providers.build_providers()``。
        传入后, 这些 provider 在 knowledge_store 之前优先查询
        (LeanSearch v2 / Loogle 等在线检索)。
        **默认 None = 与历史行为完全一致** (knowledge_store →
        本地 TF-IDF → heuristic 三级降级)。"""
        self._knowledge_store = knowledge_store
        self._premise_db_path = premise_db_path
        self._tfidf = None
        self._tfidf_init_failed = False
        self._multi_retriever = None
        if providers:
            from prover.premise.providers import MultiRetriever
            self._multi_retriever = MultiRetriever(providers=list(providers))

    def _get_tfidf(self):
        """Lazily build a TF-IDF retriever over data/premises/*.jsonl.

       
        ``TFIDFRetriever`` from ``knowledge.tfidf_retriever`` and used
        an API (constructor taking a path, ``.retrieve()`` returning
        ``(name, score)`` tuples) that never existed. The actual class
        is ``KnowledgeTFIDFRetriever``; it takes weights, exposes
        ``.index_lemmas(list[dict])`` and ``.search() -> list[ScoredLemma]``.
        Every call to this method previously fell into ``except`` and
        the tool silently degraded to the 8-pattern heuristic.
        """
        if self._tfidf is not None or self._tfidf_init_failed:
            return self._tfidf
        try:
            from knowledge.tfidf_retriever import KnowledgeTFIDFRetriever
            lemmas = self._load_premise_lemmas()
            if not lemmas:
                # Nothing to index — don't create an empty retriever
                # (treat as "no fallback available").
                self._tfidf_init_failed = True
                return None
            retriever = KnowledgeTFIDFRetriever()
            retriever.index_lemmas(lemmas)
            self._tfidf = retriever
        except Exception as exc:
            # Real failures (e.g. import error if module shape changes)
            # are now surfaced at WARNING — silent debug-level swallowing
            # is what hid this bug for ~5 versions.
            logger.warning(
                "premise_search TF-IDF init failed; falling back to "
                "heuristic search: %s", exc)
            self._tfidf_init_failed = True
        return self._tfidf

    def _load_premise_lemmas(self) -> list[dict]:
        """Load premises from data/premises/*.jsonl (same source as
        prover.premise.PremiseSelector). Each line is a JSON object
        with at least 'name' and 'statement'."""
        import glob
        import json
        import os
        candidates = []
        if self._premise_db_path:
            candidates.append(self._premise_db_path)
        # Project default
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(
            os.path.join(here, "..", "..", "..", "data", "premises"))
        candidates.append("data/premises")

        seen_files: set[str] = set()
        seen_names: set[str] = set()
        out: list[dict] = []
        for cand in candidates:
            if not cand or not os.path.isdir(cand):
                continue
            for filepath in sorted(glob.glob(os.path.join(cand, "*.jsonl"))):
                ap = os.path.realpath(filepath)
                if ap in seen_files:
                    continue
                seen_files.add(ap)
                try:
                    with open(filepath) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
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
                            })
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(
                        "premise_search: failed to load %s: %s", filepath, e)
        return out

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        query = input["query"]
        max_results = input.get("max_results", 10)
        domain = input.get("domain_filter", "")

        results = []
        degraded: list[str] = []

        # 0. 显式注入的 retriever providers (LeanSearch v2 / Loogle /
        #    leanstatesearch / local_tfidf) — 可选, 默认不存在。
        #    provider 结果带 source 字段, 与历史输出 schema 兼容。
        if self._multi_retriever is not None:
            hits, degraded = self._multi_retriever.search(
                query, top_k=max_results,
                goal_state=input.get("goal_state", ""))
            results.extend(h.to_tool_result_entry() for h in hits)

        # 1. Search knowledge store if available
        if self._knowledge_store:
            try:
                ks_results = self._knowledge_store.search(
                    query, limit=max_results, domain=domain)
                for r in ks_results:
                    results.append({
                        "name": r.get("name", ""),
                        "type_signature": r.get("type", ""),
                        "relevance": r.get("score", 0.5),
                        "source": "knowledge_store",
                    })
            except Exception as e:
                logger.debug(f"Knowledge store search failed: {e}")

        # 2. TF-IDF retriever fallback
        tfidf = self._get_tfidf()
        if tfidf and len(results) < max_results:
            try:

                # prior code called .retrieve() returning (name, score)
                # tuples — neither existed on the actual class.
                tfidf_results = tfidf.search(
                    query, top_k=max_results - len(results),
                    domain=domain or "")
                for sl in tfidf_results:
                    name = getattr(sl, "name", "")
                    if not name or any(r["name"] == name for r in results):
                        continue
                    results.append({
                        "name": name,
                        "type_signature": getattr(sl, "statement", ""),
                        "relevance": float(getattr(sl, "score", 0.0)),
                        "source": "tfidf",
                    })
            except Exception as e:
                logger.warning(f"TF-IDF search failed: {e}")

        # 3. Heuristic fallback if no backends available
        if not results:
            results = self._heuristic_search(query, max_results)
            if results:
                degraded = degraded + ["heuristic_fallback"]

        # Provider 结果保持注入顺序在前 (跨 provider 的分数不可比,
        # 不参与全局排序); 旧链路结果按 relevance 排序 — 无 provider
        # 时与历史行为逐字节一致。
        n_prov = sum(1 for r in results
                     if r.get("source") not in (
                         "knowledge_store", "tfidf", "heuristic"))
        head, tail = results[:n_prov], results[n_prov:]
        tail.sort(key=lambda r: -r.get("relevance", 0))
        results = (head + tail)[:max_results]

        return ToolResult.success(
            json.dumps(results, indent=2),
            count=len(results),
            degraded_providers=degraded,
        )

    def _heuristic_search(self, query: str, max_results: int) -> list[dict]:
        """Keyword-based fallback using common Mathlib lemma patterns."""
        common = {
            "add_comm": "theorem Nat.add_comm (n m : ℕ) : n + m = m + n",
            "add_assoc": "theorem Nat.add_assoc (a b c : ℕ) : a + b + c = a + (b + c)",
            "mul_comm": "theorem Nat.mul_comm (n m : ℕ) : n * m = m * n",
            "add_zero": "theorem Nat.add_zero (n : ℕ) : n + 0 = n",
            "zero_add": "theorem Nat.zero_add (n : ℕ) : 0 + n = n",
            "succ_add": "theorem Nat.succ_add (n m : ℕ) : n.succ + m = (n + m).succ",
            "dvd_refl": "theorem dvd_refl (a : α) : a ∣ a",
            "dvd_trans": "theorem dvd_trans {a b c : α} : a ∣ b → b ∣ c → a ∣ c",
        }
        results = []
        ql = query.lower()
        for name, sig in common.items():
            if any(word in name for word in ql.split()):
                results.append({
                    "name": name, "type_signature": sig,
                    "relevance": 0.5, "source": "heuristic",
                })
        return results[:max_results]

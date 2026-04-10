"""agent/tools/builtin/premise_search.py — Search Mathlib for relevant lemmas"""
from __future__ import annotations

import json
import logging
from typing import Optional

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
        },
        "required": ["query"],
    }

    def __init__(self, knowledge_store=None, premise_db_path: str = ""):
        self._knowledge_store = knowledge_store
        self._premise_db_path = premise_db_path
        self._tfidf = None

    def _get_tfidf(self):
        if self._tfidf is None:
            try:
                from knowledge.tfidf_retriever import TFIDFRetriever
                self._tfidf = TFIDFRetriever(self._premise_db_path)
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")
        return self._tfidf

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        query = input["query"]
        max_results = input.get("max_results", 10)
        domain = input.get("domain_filter", "")

        results = []

        # 1. Search knowledge store if available
        if self._knowledge_store:
            try:
                from knowledge.store import KnowledgeStore
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
                tfidf_results = tfidf.retrieve(
                    query, top_k=max_results - len(results))
                for name, score in tfidf_results:
                    if not any(r["name"] == name for r in results):
                        results.append({
                            "name": name,
                            "type_signature": "",
                            "relevance": float(score),
                            "source": "tfidf",
                        })
            except Exception as e:
                logger.debug(f"TF-IDF search failed: {e}")

        # 3. Heuristic fallback if no backends available
        if not results:
            results = self._heuristic_search(query, max_results)

        results.sort(key=lambda r: -r.get("relevance", 0))
        results = results[:max_results]

        return ToolResult.success(
            json.dumps(results, indent=2),
            count=len(results),
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

"""prover/premise/selector.py — 统一前提选择入口

支持三种模式: bm25, embedding, hybrid (融合两者)。

v9: 70 条 built-in 前提已 dump 到 ``data/premises/builtin_core.jsonl``,
selector 现在统一从外部 jsonl 加载, 不再保留代码内嵌副本。
"""
from __future__ import annotations
from prover.premise.bm25_retriever import BM25Retriever
from prover.premise.embedding_retriever import EmbeddingRetriever
from prover.premise.reranker import PremiseReranker


class PremiseSelector:
    """Unified premise retrieval with configurable backend.

    Modes:
        'bm25':      BM25 only
        'embedding': TF-IDF embedding only
        'hybrid':    Both, fused via reranker (RRF)
        'none':      Return empty (disabled)
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "hybrid")
        self._bm25 = BM25Retriever(
            k1=self.config.get("bm25_k1", 1.5),
            b=self.config.get("bm25_b", 0.75),
        )
        self._embed = EmbeddingRetriever()
        self._reranker = PremiseReranker()
        self._initialized = False
        self._init_lock = __import__('threading').Lock()

    def _ensure_init(self):
        if self._initialized:
            return
        with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            # v9: load all premises from data/premises/*.jsonl
            # (builtin_core.jsonl + mathlib_core.jsonl + any user files).
            all_premises = self._load_external_premises()
            seen = {p["name"] for p in all_premises}

            # Extra premises from config (highest priority — overrides nothing,
            # just adds names not already covered).
            extra = self.config.get("extra_premises", [])
            for p in extra:
                if p.get("name") and p["name"] not in seen:
                    all_premises.append(p)
                    seen.add(p["name"])

            self._bm25.add_documents(all_premises)
            self._embed.add_documents(all_premises)
            self._bm25.build()
            self._embed.build()
            self._initialized = True
            self._premise_count = len(all_premises)

            import logging
            logging.getLogger(__name__).info(
                f"PremiseSelector initialized with "
                f"{len(all_premises)} premises (loaded from data/premises/*.jsonl)")

    def _load_external_premises(self) -> list[dict]:
        """Load premises from data/premises/*.jsonl files."""
        import json
        import os
        import glob

        premises = []
        # Search multiple locations (dedup so the same dir isn't visited twice).
        search_dirs = self.config.get("premise_dirs", [
            "data/premises",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "premises"),
        ])

        seen_files: set[str] = set()
        seen_names: set[str] = set()
        for search_dir in search_dirs:
            pattern = os.path.join(search_dir, "*.jsonl")
            for filepath in sorted(glob.glob(pattern)):
                abs_path = os.path.realpath(filepath)
                if abs_path in seen_files:
                    continue
                seen_files.add(abs_path)
                try:
                    with open(filepath) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            if "name" not in entry or "statement" not in entry:
                                continue
                            name = entry["name"]
                            if name in seen_names:
                                continue  # First file wins (jsonl sorted alphabetically).
                            seen_names.add(name)
                            premises.append(entry)
                except (json.JSONDecodeError, OSError) as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to load premise file {filepath}: {e}")

        return premises

    def add_premises(self, premises: list[dict]):
        """Dynamically add premises at runtime."""
        self._bm25.add_documents(premises)
        self._embed.add_documents(premises)
        self._bm25.build()
        self._embed.build()

    def load_from_mathlib_export(self, filepath: str) -> int:
        """从 scripts/export_mathlib_premises.py 的输出加载完整 Mathlib 前提库.

        用法::

            selector = PremiseSelector()
            count = selector.load_from_mathlib_export("data/premises/mathlib_full.jsonl")
            print(f"Loaded {count} Mathlib premises")

        文件格式: 每行一个 JSON 对象, 含 "name" 和 "statement" 字段。
        支持增量加载: 可多次调用, 自动去重。

        Returns:
            新增的前提数量
        """
        import json as _json

        self._ensure_init()
        existing = set()
        try:
            # 收集已有名称用于去重
            for doc in self._bm25._documents:
                existing.add(doc.get("name", ""))
        except (AttributeError, TypeError) as _exc:
            logger.debug(f"Suppressed exception: {_exc}")

        new_premises = []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = _json.loads(line)
                    name = entry.get("name", "")
                    if name and name not in existing:
                        new_premises.append(entry)
                        existing.add(name)
        except (OSError, _json.JSONDecodeError) as e:
            import logging
            logging.getLogger(__name__).error(
                f"Failed to load Mathlib export from {filepath}: {e}")
            return 0

        if new_premises:
            self.add_premises(new_premises)
            import logging
            logging.getLogger(__name__).info(
                f"Loaded {len(new_premises)} new premises from {filepath} "
                f"(total: {len(existing)})")

        return len(new_premises)

    def retrieve(self, theorem: str, top_k: int = 10,
                 goal_type: str = "", tactic_hint: str = "") -> list[dict]:
        """Retrieve relevant premises for a theorem statement."""
        self._ensure_init()

        if self.mode == "none":
            return []

        if self.mode == "bm25":
            return self._bm25.retrieve(theorem, top_k)

        if self.mode == "embedding":
            return self._embed.retrieve(theorem, top_k)

        # Hybrid: retrieve from both, merge via reranker
        bm25_results = self._bm25.retrieve(theorem, top_k * 2)
        embed_results = self._embed.retrieve(theorem, top_k * 2)

        # Deduplicate by name, keeping best score
        merged: dict[str, dict] = {}
        for r in bm25_results + embed_results:
            name = r["name"]
            if name not in merged or r["score"] > merged[name]["score"]:
                merged[name] = r
        candidates = list(merged.values())

        return self._reranker.rerank(candidates, theorem, goal_type,
                                      tactic_hint, top_k)

    @property
    def size(self) -> int:
        self._ensure_init()
        return getattr(self, '_premise_count', self._bm25.size)

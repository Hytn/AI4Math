"""prover/premise/selector.py — 统一前提选择入口

支持三种模式: bm25, embedding, hybrid (融合两者)。
"""
from __future__ import annotations
from prover.premise.bm25_retriever import BM25Retriever
from prover.premise.embedding_retriever import EmbeddingRetriever
from prover.premise.reranker import PremiseReranker


# Built-in Mathlib premise library (common lemmas for bootstrapping)
# Organized by category for maintainability
_BUILTIN_PREMISES = [
    # ── Natural numbers ──
    {"name": "Nat.add_comm", "statement": "theorem Nat.add_comm (n m : Nat) : n + m = m + n"},
    {"name": "Nat.add_assoc", "statement": "theorem Nat.add_assoc (n m k : Nat) : n + m + k = n + (m + k)"},
    {"name": "Nat.mul_comm", "statement": "theorem Nat.mul_comm (n m : Nat) : n * m = m * n"},
    {"name": "Nat.mul_assoc", "statement": "theorem Nat.mul_assoc (n m k : Nat) : n * m * k = n * (m * k)"},
    {"name": "Nat.add_zero", "statement": "theorem Nat.add_zero (n : Nat) : n + 0 = n"},
    {"name": "Nat.zero_add", "statement": "theorem Nat.zero_add (n : Nat) : 0 + n = n"},
    {"name": "Nat.mul_zero", "statement": "theorem Nat.mul_zero (n : Nat) : n * 0 = 0"},
    {"name": "Nat.zero_mul", "statement": "theorem Nat.zero_mul (n : Nat) : 0 * n = 0"},
    {"name": "Nat.mul_one", "statement": "theorem Nat.mul_one (n : Nat) : n * 1 = n"},
    {"name": "Nat.one_mul", "statement": "theorem Nat.one_mul (n : Nat) : 1 * n = n"},
    {"name": "Nat.succ_eq_add_one", "statement": "theorem Nat.succ_eq_add_one (n : Nat) : n.succ = n + 1"},
    {"name": "Nat.add_succ", "statement": "theorem Nat.add_succ (n m : Nat) : n + m.succ = (n + m).succ"},
    {"name": "Nat.succ_add", "statement": "theorem Nat.succ_add (n m : Nat) : n.succ + m = (n + m).succ"},
    {"name": "Nat.le_refl", "statement": "theorem Nat.le_refl (n : Nat) : n ≤ n"},
    {"name": "Nat.le_trans", "statement": "theorem Nat.le_trans {a b c : Nat} : a ≤ b → b ≤ c → a ≤ c"},
    {"name": "Nat.lt_of_lt_of_le", "statement": "theorem Nat.lt_of_lt_of_le {a b c : Nat} : a < b → b ≤ c → a < c"},
    {"name": "Nat.le_of_lt", "statement": "theorem Nat.le_of_lt {a b : Nat} : a < b → a ≤ b"},
    {"name": "Nat.sub_self", "statement": "theorem Nat.sub_self (n : Nat) : n - n = 0"},
    {"name": "Nat.add_left_cancel", "statement": "theorem Nat.add_left_cancel {a b c : Nat} (h : a + b = a + c) : b = c"},
    {"name": "Nat.mul_add", "statement": "theorem Nat.mul_add (a b c : Nat) : a * (b + c) = a * b + a * c"},
    {"name": "Nat.add_mul", "statement": "theorem Nat.add_mul (a b c : Nat) : (a + b) * c = a * c + b * c"},
    {"name": "Nat.pow_succ", "statement": "theorem Nat.pow_succ (a n : Nat) : a ^ (n + 1) = a ^ n * a"},
    {"name": "Nat.pos_of_ne_zero", "statement": "theorem Nat.pos_of_ne_zero {n : Nat} (h : n ≠ 0) : 0 < n"},

    # ── Integers ──
    {"name": "Int.add_comm", "statement": "theorem Int.add_comm (a b : Int) : a + b = b + a"},
    {"name": "Int.mul_comm", "statement": "theorem Int.mul_comm (a b : Int) : a * b = b * a"},
    {"name": "Int.add_assoc", "statement": "theorem Int.add_assoc (a b c : Int) : a + b + c = a + (b + c)"},
    {"name": "Int.neg_neg", "statement": "theorem Int.neg_neg (a : Int) : -(-a) = a"},
    {"name": "Int.add_neg_cancel", "statement": "theorem Int.add_neg_cancel (a : Int) : a + (-a) = 0"},
    {"name": "abs_nonneg", "statement": "theorem abs_nonneg (a : Int) : 0 ≤ |a|"},
    {"name": "abs_mul", "statement": "theorem abs_mul (a b : Int) : |a * b| = |a| * |b|"},

    # ── Logic ──
    {"name": "And.intro", "statement": "theorem And.intro {a b : Prop} (ha : a) (hb : b) : a ∧ b"},
    {"name": "And.left", "statement": "theorem And.left {a b : Prop} (h : a ∧ b) : a"},
    {"name": "And.right", "statement": "theorem And.right {a b : Prop} (h : a ∧ b) : b"},
    {"name": "Or.inl", "statement": "theorem Or.inl {a b : Prop} (h : a) : a ∨ b"},
    {"name": "Or.inr", "statement": "theorem Or.inr {a b : Prop} (h : b) : a ∨ b"},
    {"name": "Or.elim", "statement": "theorem Or.elim {a b c : Prop} (h : a ∨ b) (ha : a → c) (hb : b → c) : c"},
    {"name": "not_not", "statement": "theorem not_not {a : Prop} [Decidable a] : ¬¬a ↔ a"},
    {"name": "Classical.em", "statement": "theorem Classical.em (p : Prop) : p ∨ ¬p"},
    {"name": "Classical.byContradiction", "statement": "theorem Classical.byContradiction {p : Prop} (h : ¬p → False) : p"},
    {"name": "Iff.intro", "statement": "theorem Iff.intro {a b : Prop} (hab : a → b) (hba : b → a) : a ↔ b"},

    # ── Equality ──
    {"name": "Eq.symm", "statement": "theorem Eq.symm {a b : α} (h : a = b) : b = a"},
    {"name": "Eq.trans", "statement": "theorem Eq.trans {a b c : α} (h1 : a = b) (h2 : b = c) : a = c"},
    {"name": "congrArg", "statement": "theorem congrArg {a b : α} (f : α → β) (h : a = b) : f a = f b"},
    {"name": "congrFun", "statement": "theorem congrFun {f g : α → β} (h : f = g) (a : α) : f a = g a"},

    # ── Real numbers ──
    {"name": "Real.add_comm", "statement": "theorem Real.add_comm (a b : ℝ) : a + b = b + a"},
    {"name": "Real.mul_comm", "statement": "theorem Real.mul_comm (a b : ℝ) : a * b = b * a"},
    {"name": "sq_nonneg", "statement": "theorem sq_nonneg (a : α) : 0 ≤ a ^ 2"},
    {"name": "sq_abs", "statement": "theorem sq_abs (a : α) : |a| ^ 2 = a ^ 2"},
    {"name": "sub_sq", "statement": "theorem sub_sq (a b : α) : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2"},
    {"name": "add_sq", "statement": "theorem add_sq (a b : α) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2"},
    {"name": "mul_self_nonneg", "statement": "theorem mul_self_nonneg (a : α) : 0 ≤ a * a"},
    {"name": "div_add_div_same", "statement": "theorem div_add_div_same (a b c : α) (hc : c ≠ 0) : a / c + b / c = (a + b) / c"},
    {"name": "mul_div_cancel'", "statement": "theorem mul_div_cancel' (a : α) {b : α} (hb : b ≠ 0) : a * b / b = a"},

    # ── Finset / Combinatorics ──
    {"name": "Finset.sum_range_succ", "statement": "theorem Finset.sum_range_succ {f : Nat → α} {n : Nat} : (Finset.range (n+1)).sum f = (Finset.range n).sum f + f n"},
    {"name": "Finset.sum_empty", "statement": "theorem Finset.sum_empty {f : α → β} : (∅ : Finset α).sum f = 0"},
    {"name": "Finset.card_range", "statement": "theorem Finset.card_range (n : Nat) : (Finset.range n).card = n"},
    {"name": "Finset.mem_range", "statement": "theorem Finset.mem_range {n k : Nat} : k ∈ Finset.range n ↔ k < n"},

    # ── List ──
    {"name": "List.length_nil", "statement": "theorem List.length_nil : ([] : List α).length = 0"},
    {"name": "List.length_cons", "statement": "theorem List.length_cons (a : α) (l : List α) : (a :: l).length = l.length + 1"},
    {"name": "List.map_nil", "statement": "theorem List.map_nil (f : α → β) : List.map f [] = []"},
    {"name": "List.append_nil", "statement": "theorem List.append_nil (l : List α) : l ++ [] = l"},

    # ── Functions ──
    {"name": "Function.comp_id", "statement": "theorem Function.comp_id (f : α → β) : f ∘ id = f"},
    {"name": "Function.id_comp", "statement": "theorem Function.id_comp (f : α → β) : id ∘ f = f"},
    {"name": "Function.Injective.comp", "statement": "theorem Function.Injective.comp {f : β → γ} {g : α → β} (hf : Function.Injective f) (hg : Function.Injective g) : Function.Injective (f ∘ g)"},

    # ── Divisibility / Number theory ──
    {"name": "Nat.dvd_refl", "statement": "theorem Nat.dvd_refl (n : Nat) : n ∣ n"},
    {"name": "Nat.dvd_trans", "statement": "theorem Nat.dvd_trans {a b c : Nat} : a ∣ b → b ∣ c → a ∣ c"},
    {"name": "Nat.dvd_add", "statement": "theorem Nat.dvd_add {a b c : Nat} : a ∣ b → a ∣ c → a ∣ (b + c)"},
    {"name": "Nat.dvd_mul_left", "statement": "theorem Nat.dvd_mul_left (a b : Nat) : a ∣ b * a"},
    {"name": "Nat.dvd_mul_right", "statement": "theorem Nat.dvd_mul_right (a b : Nat) : a ∣ a * b"},
    {"name": "Nat.gcd_comm", "statement": "theorem Nat.gcd_comm (a b : Nat) : Nat.gcd a b = Nat.gcd b a"},
]


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

            # 1. Start with built-in premises (backward compat)
            all_premises = list(_BUILTIN_PREMISES)

            # 2. Load external premise files (data/premises/*.jsonl)
            external = self._load_external_premises()
            if external:
                # Deduplicate by name (external overrides built-in)
                seen = {p["name"] for p in all_premises}
                for p in external:
                    if p["name"] not in seen:
                        all_premises.append(p)
                        seen.add(p["name"])

            # 3. Add any extra premises from config
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
                f"PremiseSelector initialized with {len(all_premises)} premises "
                f"({len(_BUILTIN_PREMISES)} built-in + {len(external)} external)")

    def _load_external_premises(self) -> list[dict]:
        """Load premises from data/premises/*.jsonl files."""
        import json
        import os
        import glob

        premises = []
        # Search multiple locations
        search_dirs = self.config.get("premise_dirs", [
            "data/premises",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "premises"),
        ])

        for search_dir in search_dirs:
            pattern = os.path.join(search_dir, "*.jsonl")
            for filepath in sorted(glob.glob(pattern)):
                try:
                    with open(filepath) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entry = json.loads(line)
                                # Normalize: ensure 'name' and 'statement' exist
                                if "name" in entry and "statement" in entry:
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
        except (AttributeError, TypeError):
            pass

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

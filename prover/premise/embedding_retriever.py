"""prover/premise/embedding_retriever.py — Sparse TF-IDF + char-n-gram retriever

⚠️  **Naming caveat**: this module is *not* a neural / dense
    embedding retriever despite the legacy filename. It is **pure
    sparse TF-IDF** over two parallel feature spaces:

      1. **Word-level TF-IDF** — captures exact keyword matches
         (e.g. "Nat.add_comm" matches query "add_comm")
      2. **Character n-gram TF-IDF** (n=3,4) — captures sub-token
         similarity (e.g. "comm" in "Nat.mul_comm" matches "commutative")

    The two scores are fused via weighted sum (configurable). This
    significantly outperforms pure word TF-IDF on Lean identifiers
    where camelCase, dot-notation, and abbreviations are common, but it
    is **categorically not** what ReProver-style retrieval means in the
    literature — that's dense neural retrieval (ColBERT / SBERT).

    Effects on evaluation:
      - The ``--profile reprover`` profile claims "ReProver-style RAG"
        but with this module is missing ReProver's actual contribution
        (a fine-tuned dense retriever over Mathlib). The retrieval
        component is closer to a Lean-aware BM25 fork.
      - On the bundled ``data/premises/mathlib_core.jsonl`` which has
        only ~334 entries (Mathlib4 itself has ~10⁵), recall is
        physically capped low regardless of which retriever is used.
        Real eval should expand the premise pool first via
        ``scripts/export_mathlib_premises.py``.

    To plug in a real dense retriever, replace ``_vectorize`` /
    ``_score`` with ``sentence-transformers/all-MiniLM-L6-v2`` (or
    LeanDojo's BYTE5-finetuned retriever) and add ``sentence-transformers``
    to ``requirements.txt`` — currently it's a commented-out optional dep.

For backward compatibility ``EmbeddingRetriever`` keeps its name; the
honest alias ``TfidfNgramRetriever`` is exported below.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from prover.premise.bm25_retriever import tokenize

def _char_ngrams(text: str, ns: tuple[int, ...] = (3, 4)) -> list[str]:
    """Extract character n-grams from text.

    Lean-aware: splits on dots and underscores first, then extracts
    n-grams from each segment.  This means "Nat.add_comm" produces
    n-grams for "nat", "add", "comm" separately, improving precision.
    """
    text = text.lower()
    segments = re.split(r'[._\s]+', text)
    segments = [s for s in segments if len(s) >= 2]
    expanded = []
    for seg in segments:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', seg).split()
        expanded.extend(p.lower() for p in parts if len(p) >= 2)

    ngrams = []
    for seg in expanded:
        padded = f"${seg}$"
        for n in ns:
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i:i + n])
    return ngrams

@dataclass
class IndexedDoc:
    name: str
    statement: str
    doc_type: str = "lemma"
    word_vec: dict[str, float] = field(default_factory=dict)
    word_norm: float = 0.0
    ngram_vec: dict[str, float] = field(default_factory=dict)
    ngram_norm: float = 0.0

class EmbeddingRetriever:
    """Hybrid word + character n-gram retriever.

    Fuses word-level and character n-gram TF-IDF scores for robust
    matching of Lean4 identifiers and mathematical terms.
    """

    def __init__(self, word_weight: float = 0.4, ngram_weight: float = 0.6,
                 ngram_sizes: tuple[int, ...] = (3, 4)):
        self.word_weight = word_weight
        self.ngram_weight = ngram_weight
        self.ngram_sizes = ngram_sizes
        self.documents: list[IndexedDoc] = []
        self._word_idf: dict[str, float] = {}
        self._ngram_idf: dict[str, float] = {}
        self._built = False

    def add_document(self, name: str, statement: str,
                     doc_type: str = "lemma"):
        self.documents.append(IndexedDoc(
            name=name, statement=statement, doc_type=doc_type))
        self._built = False

    def add_documents(self, docs: list[dict]):
        for d in docs:
            self.add_document(d["name"], d["statement"],
                              d.get("doc_type", "lemma"))

    def build(self):
        """Build both word-level and n-gram TF-IDF indices."""
        N = len(self.documents)
        if N == 0:
            self._built = True
            return

        word_df: Counter = Counter()
        ngram_df: Counter = Counter()
        all_word_tokens = []
        all_ngram_tokens = []

        for doc in self.documents:
            text = f"{doc.name} {doc.statement}"
            words = tokenize(text)
            ngrams = _char_ngrams(text, self.ngram_sizes)
            all_word_tokens.append(words)
            all_ngram_tokens.append(ngrams)
            for t in set(words):
                word_df[t] += 1
            for t in set(ngrams):
                ngram_df[t] += 1

        self._word_idf = {
            t: math.log((N + 1) / (f + 1)) + 1.0
            for t, f in word_df.items()
        }
        self._ngram_idf = {
            t: math.log((N + 1) / (f + 1)) + 1.0
            for t, f in ngram_df.items()
        }

        for doc, words, ngrams in zip(self.documents, all_word_tokens,
                                       all_ngram_tokens):
            wtf = Counter(words)
            wt = len(words) or 1
            doc.word_vec = {
                t: (c / wt) * self._word_idf.get(t, 1.0)
                for t, c in wtf.items()
            }
            doc.word_norm = math.sqrt(
                sum(v * v for v in doc.word_vec.values())) or 1.0

            ntf = Counter(ngrams)
            nt = len(ngrams) or 1
            doc.ngram_vec = {
                t: (c / nt) * self._ngram_idf.get(t, 1.0)
                for t, c in ntf.items()
            }
            doc.ngram_norm = math.sqrt(
                sum(v * v for v in doc.ngram_vec.values())) or 1.0

        self._built = True

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve most relevant premises using hybrid scoring."""
        if not self._built:
            self.build()
        if not self.documents:
            return []

        words = tokenize(query)
        ngrams = _char_ngrams(query, self.ngram_sizes)
        if not words and not ngrams:
            return []

        wtf = Counter(words)
        wt = len(words) or 1
        wq = {t: (c / wt) * self._word_idf.get(t, 1.0)
              for t, c in wtf.items()}
        wq_norm = math.sqrt(sum(v * v for v in wq.values())) or 1.0

        ntf = Counter(ngrams)
        nt = len(ngrams) or 1
        nq = {t: (c / nt) * self._ngram_idf.get(t, 1.0)
              for t, c in ntf.items()}
        nq_norm = math.sqrt(sum(v * v for v in nq.values())) or 1.0

        scored = []
        for doc in self.documents:
            w_dot = sum(wq.get(t, 0) * doc.word_vec.get(t, 0)
                        for t in wq if t in doc.word_vec)
            w_sim = (w_dot / (wq_norm * doc.word_norm)
                     if wq_norm and doc.word_norm else 0.0)

            n_dot = sum(nq.get(t, 0) * doc.ngram_vec.get(t, 0)
                        for t in nq if t in doc.ngram_vec)
            n_sim = (n_dot / (nq_norm * doc.ngram_norm)
                     if nq_norm and doc.ngram_norm else 0.0)

            score = self.word_weight * w_sim + self.ngram_weight * n_sim
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        return [
            {"name": doc.name, "statement": doc.statement,
             "score": round(score, 4), "doc_type": doc.doc_type}
            for score, doc in scored[:top_k]
        ]

    @property
    def size(self) -> int:
        return len(self.documents)

# so the call site reads as "I'm using sparse TF-IDF, not a neural
# embedding retriever". The old name is kept indefinitely for
# backward compatibility — multiple callers in ``prover/premise/selector.py``
# and tests still reference it.
TfidfNgramRetriever = EmbeddingRetriever

__all__ = ["EmbeddingRetriever", "TfidfNgramRetriever", "IndexedDoc"]

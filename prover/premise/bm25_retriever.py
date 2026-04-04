"""prover/premise/bm25_retriever.py — BM25 前提检索

实现 Okapi BM25 算法，用于从 Mathlib 定理库中检索与当前目标相关的前提。
不依赖外部库，纯 Python 实现。
"""
from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Document:
    """A retrievable premise document."""
    name: str
    statement: str
    doc_type: str = "lemma"
    tokens: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Lean4-aware tokenizer: splits on whitespace, punctuation, and camelCase.
    
    Keeps single-letter tokens if they look like math variables (a-z, A-Z)
    since these are common in theorem statements (n, m, P, Q, etc.).
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    tokens = re.findall(r'[A-Za-z][a-z]*|[A-Z]|[0-9]+', text)
    return [t.lower() for t in tokens if t]


class BM25Retriever:
    """Okapi BM25 retriever for Lean4 premises.

    Parameters:
        k1: Term frequency saturation parameter (default 1.5)
        b:  Length normalization parameter (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self._doc_freqs: Counter = Counter()
        self._avg_dl: float = 0.0
        self._N: int = 0
        self._built = False

    def add_document(self, name: str, statement: str, doc_type: str = "lemma",
                     metadata: dict = None):
        tokens = tokenize(f"{name} {statement}")
        doc = Document(name=name, statement=statement, doc_type=doc_type,
                       tokens=tokens, metadata=metadata or {})
        self.documents.append(doc)
        self._built = False

    def add_documents(self, docs: list[dict]):
        for d in docs:
            self.add_document(d["name"], d["statement"],
                              d.get("doc_type", "lemma"), d.get("metadata"))

    def build(self):
        self._N = len(self.documents)
        if self._N == 0:
            self._built = True
            return
        total_len = 0
        self._doc_freqs = Counter()
        for doc in self.documents:
            total_len += len(doc.tokens)
            for term in set(doc.tokens):
                self._doc_freqs[term] += 1
        self._avg_dl = total_len / self._N
        self._built = True

    def _idf(self, term: str) -> float:
        df = self._doc_freqs.get(term, 0)
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, query_tokens: list[str], doc: Document) -> float:
        dl = len(doc.tokens)
        tf_map = Counter(doc.tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            idf = self._idf(qt)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * numerator / denominator
        return score

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._built:
            self.build()
        if self._N == 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored = []
        for doc in self.documents:
            s = self._score_doc(query_tokens, doc)
            if s > 0:
                scored.append((s, doc))
        scored.sort(key=lambda x: -x[0])
        return [
            {"name": doc.name, "statement": doc.statement,
             "score": round(score, 4), "doc_type": doc.doc_type}
            for score, doc in scored[:top_k]
        ]

    @property
    def size(self) -> int:
        return len(self.documents)

"""knowledge/tfidf_retriever.py — TF-IDF 增强的知识检索

替代 knowledge/store.py 中的纯关键词重叠检索。
复用 prover/premise/embedding_retriever.py 中的 char n-gram TF-IDF 方案。

关键改进：
  1. char n-gram (3-5) 捕捉 Lean4 标识符的子词相似性
  2. 对 goal pattern 做语义相似度排序（不仅仅 set intersection）
  3. 对 lemma statement 做全文 TF-IDF 检索
  4. 融合 BM25 关键词匹配 + TF-IDF 向量相似度

Usage::

    enhancer = KnowledgeTFIDFRetriever()
    enhancer.index_lemmas(lemma_records)
    results = enhancer.search("⊢ n + 0 = n", top_k=5)
"""
from __future__ import annotations

import math
import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _tokenize_lean(text: str) -> list[str]:
    """Lean4 专用分词器：保留类型名/策略名/符号"""
    # 分词: 驼峰边界、点号、空格、运算符
    tokens = re.findall(r'[A-Z][a-z]+|[a-z_]\w*|[A-Z]+|[→∀∃∧∨¬⊢≤≥≠∣⟨⟩]|[+\-*/=<>]|\d+', text)
    return [t.lower() for t in tokens if len(t) > 1 or t in '→∀∃∧∨¬⊢≤≥']


def _char_ngrams(text: str, ns: tuple[int, ...] = (3, 4, 5)) -> list[str]:
    """提取 char n-grams"""
    text = text.lower().strip()
    grams = []
    for n in ns:
        for i in range(len(text) - n + 1):
            grams.append(text[i:i + n])
    return grams


@dataclass
class ScoredLemma:
    name: str
    statement: str
    proof: str
    score: float
    match_reason: str = ""


class KnowledgeTFIDFRetriever:
    """TF-IDF + BM25 融合的知识检索引擎。

    索引建立在 lemma statement 上。查询时将 goal 文本
    与所有 lemma 计算相似度，融合关键词匹配和 n-gram 相似度。
    """

    def __init__(self, bm25_weight: float = 0.4, tfidf_weight: float = 0.6):
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight
        # 索引数据
        self._docs: list[dict] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_ngrams: list[list[str]] = []
        # BM25 参数
        self._avg_dl: float = 0.0
        self._df: Counter = Counter()
        self._N: int = 0
        # TF-IDF (char n-gram)
        self._ngram_df: Counter = Counter()
        self._indexed: bool = False

    def index_lemmas(self, lemmas: list[dict]):
        """构建索引。每个 lemma 是 dict with 'name', 'statement', 'proof'."""
        self._docs = list(lemmas)
        self._N = len(lemmas)
        if self._N == 0:
            return

        self._doc_tokens = []
        self._doc_ngrams = []
        self._df = Counter()
        self._ngram_df = Counter()

        for doc in lemmas:
            text = f"{doc.get('name', '')} {doc.get('statement', '')}"
            tokens = _tokenize_lean(text)
            ngrams = _char_ngrams(text)
            self._doc_tokens.append(tokens)
            self._doc_ngrams.append(ngrams)
            # DF
            for t in set(tokens):
                self._df[t] += 1
            for g in set(ngrams):
                self._ngram_df[g] += 1

        self._avg_dl = sum(len(t) for t in self._doc_tokens) / max(1, self._N)
        self._indexed = True
        logger.info(f"Indexed {self._N} lemmas for TF-IDF retrieval")

    def search(self, query: str, top_k: int = 10,
               domain: str = "", goal_pattern: str = "") -> list[ScoredLemma]:
        """融合 BM25 + TF-IDF 检索。"""
        if not self._indexed or self._N == 0:
            return []

        full_query = f"{query} {goal_pattern}"
        q_tokens = _tokenize_lean(full_query)
        q_ngrams = _char_ngrams(full_query)

        scores = []
        for i, doc in enumerate(self._docs):
            bm25 = self._bm25_score(q_tokens, i)
            tfidf = self._ngram_tfidf_score(q_ngrams, i)
            combined = self.bm25_weight * bm25 + self.tfidf_weight * tfidf

            # Domain boost
            if domain and doc.get('domain', '') == domain:
                combined *= 1.3

            # Citation boost (mild)
            citations = doc.get('times_cited', 0)
            combined += min(citations * 0.05, 0.5)

            scores.append((combined, i))

        scores.sort(key=lambda x: -x[0])

        results = []
        for score, idx in scores[:top_k]:
            doc = self._docs[idx]
            results.append(ScoredLemma(
                name=doc.get('name', ''),
                statement=doc.get('statement', ''),
                proof=doc.get('proof', ''),
                score=score,
                match_reason=f"bm25+tfidf={score:.3f}",
            ))
        return results

    def _bm25_score(self, query_tokens: list[str], doc_idx: int,
                     k1: float = 1.5, b: float = 0.75) -> float:
        """Okapi BM25"""
        doc_tokens = self._doc_tokens[doc_idx]
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in self._df:
                continue
            df = self._df[qt]
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            term_tf = tf.get(qt, 0)
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * dl / max(1, self._avg_dl))
            score += idf * numerator / denominator
        return score

    def _ngram_tfidf_score(self, query_ngrams: list[str], doc_idx: int) -> float:
        """Char n-gram TF-IDF cosine similarity"""
        doc_ngrams = self._doc_ngrams[doc_idx]
        if not doc_ngrams or not query_ngrams:
            return 0.0

        doc_tf = Counter(doc_ngrams)
        q_tf = Counter(query_ngrams)

        # Compute dot product with IDF weighting
        dot = 0.0
        q_norm = 0.0
        d_norm = 0.0

        all_grams = set(q_tf) | set(doc_tf)
        for g in all_grams:
            df = max(1, self._ngram_df.get(g, 1))
            idf = math.log(self._N / df + 1.0)
            q_val = q_tf.get(g, 0) * idf
            d_val = doc_tf.get(g, 0) * idf
            dot += q_val * d_val
            q_norm += q_val ** 2
            d_norm += d_val ** 2

        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))


def enhance_knowledge_store_search(store, query_goal: str, query_theorem: str = "",
                                    domain: str = "", top_k: int = 10) -> list[ScoredLemma]:
    """便捷函数：从 UnifiedKnowledgeStore 提取 lemma，用 TF-IDF 重排序。

    替代 store._search_lemmas_sync 的纯 keyword overlap 检索。
    """
    import json, sqlite3

    retriever = KnowledgeTFIDFRetriever()

    # 从 store 的 SQLite 中获取 lemma 候选
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT name, statement, proof, keywords, domain, times_cited, decay_factor "
            "FROM proved_lemmas WHERE stale=0 "
            "ORDER BY (times_cited * decay_factor) DESC LIMIT ?",
            (top_k * 5,)  # over-fetch for re-ranking
        ).fetchall()

    lemmas = []
    for r in rows:
        lemmas.append({
            "name": r["name"], "statement": r["statement"],
            "proof": r["proof"], "domain": r["domain"],
            "times_cited": r["times_cited"],
            "keywords": json.loads(r["keywords"] or "[]"),
        })

    if not lemmas:
        return []

    retriever.index_lemmas(lemmas)
    return retriever.search(
        f"{query_theorem} {query_goal}", top_k=top_k, domain=domain,
        goal_pattern=query_goal)

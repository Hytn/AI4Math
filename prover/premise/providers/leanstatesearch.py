"""prover/premise/providers/leanstatesearch.py — LeanStateSearch provider

LeanStateSearch (https://premise-search.com, Tao et al. 2025) 是
**proof-state-conditioned** 的前提检索: 输入不是自然语言查询而是
Lean 4 REPL 的形式化证明状态。在 step-level profile (leandojo /
best_first / mcts) 里它比语义检索更对口 —— 这是它与
leansearch_v2 / loogle 的本质分工差异。

调用约定: ``search(query, goal_state=...)`` — 优先用 goal_state,
缺省时退回 query (有些调用方只有 NL 查询; 此时效果会打折, 记 INFO)。

⚠️ 标注 experimental: premise-search.com 的公开 API schema 未冻结,
   字段映射做了容错; endpoint 经 AI4MATH_LEANSTATESEARCH_URL 覆盖。
   请求格式: POST {"query": <state>, "results": k, "rev": <mathlib rev>}
   (rev 经 AI4MATH_LEANSTATESEARCH_REV 配置, 默认留空由服务端取最新。)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from prover.premise.providers.base import (
    RetrievedPremise, RetrieverProvider, register_provider)
from prover.premise.providers.cache import RetrievalCache

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://premise-search.com/api/search"
_UA = "AI4Math-retriever/1.0"


@register_provider
class LeanStateSearchProvider(RetrieverProvider):
    name = "leanstatesearch"

    def __init__(self, url: str = "", timeout: float = 0.0,
                 cache: RetrievalCache | None = None, **_):
        self.url = url or os.environ.get(
            "AI4MATH_LEANSTATESEARCH_URL", _DEFAULT_URL)
        self.rev = os.environ.get("AI4MATH_LEANSTATESEARCH_REV", "")
        self.timeout = timeout or float(
            os.environ.get("AI4MATH_LEANSTATESEARCH_TIMEOUT", "10"))
        self.cache = cache if cache is not None else RetrievalCache.from_env()
        self._consecutive_failures = 0

    def available(self) -> bool:
        return self._consecutive_failures < 3

    def search(self, query: str, top_k: int = 10, *,
               goal_state: str = "") -> list[RetrievedPremise]:
        state = (goal_state or "").strip()
        if not state:
            state = (query or "").strip()
            if state:
                logger.info(
                    "leanstatesearch: no goal_state given, falling back to "
                    "NL query — retrieval quality will degrade (this "
                    "retriever is state-conditioned).")
        if not state:
            return []

        # 缓存键与 top_k 解耦 (见 leansearch_v2 同处注释)
        key = RetrievalCache.make_key(
            self.name, state, 0, extra=f"{self.url}|{self.rev}")
        cached = self.cache.get(key)
        if cached is not None:
            return [RetrievedPremise(source=self.name, **d)
                    for d in cached][:top_k]
        if self.cache.mode == "ro":
            logger.warning("leanstatesearch: cache miss in ro mode")
            return []

        body: dict = {"query": state, "results": max(top_k, 10)}
        if self.rev:
            body["rev"] = self.rev
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = json.loads(r.read().decode())
            self._consecutive_failures = 0
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as e:
            self._consecutive_failures += 1
            logger.warning("leanstatesearch unreachable: %s", e)
            return []

        hits = raw
        if isinstance(raw, dict):
            hits = raw.get("results") or raw.get("premises") or []
        if not isinstance(hits, list):
            logger.warning("leanstatesearch: unrecognised response shape")
            return []
        out = []
        for i, h in enumerate(hits[:max(top_k, 10)]):
            if not isinstance(h, dict):
                continue
            name = h.get("name") or h.get("formal_name") or ""
            if not name:
                continue
            score = h.get("score")
            out.append(RetrievedPremise(
                name=str(name),
                statement=str(h.get("statement") or h.get("type") or ""),
                score=float(score) if score is not None else 1.0 - i * 0.01,
                source=self.name,
                module=str(h.get("module") or ""),
            ))
        self.cache.put(key, state, [
            {"name": p.name, "statement": p.statement, "score": p.score,
             "module": p.module} for p in out])
        return out[:top_k]

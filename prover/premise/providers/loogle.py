"""prover/premise/providers/loogle.py — Loogle 模式匹配检索 provider

Loogle (https://loogle.lean-lang.org) 是按**类型签名/模式**匹配的检索器
(区别于 LeanSearch 的语义检索)。两者互补: Loogle 擅长
"⊢ _ * (_ + _) = _" 这种结构化模式, LeanSearch 擅长自然语言意图。

API: GET https://loogle.lean-lang.org/json?q=<pattern>
返回 {"hits": [{"name", "module", "type", "doc"}], "error": ...?}

环境变量: AI4MATH_LOOGLE_URL / AI4MATH_LOOGLE_TIMEOUT
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from prover.premise.providers.base import (
    RetrievedPremise, RetrieverProvider, register_provider)
from prover.premise.providers.cache import RetrievalCache

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://loogle.lean-lang.org/json"
_UA = "AI4Math-retriever/1.0"


@register_provider
class LoogleProvider(RetrieverProvider):
    name = "loogle"

    def __init__(self, url: str = "", timeout: float = 0.0,
                 cache: RetrievalCache | None = None, **_):
        self.url = url or os.environ.get("AI4MATH_LOOGLE_URL", _DEFAULT_URL)
        self.timeout = timeout or float(
            os.environ.get("AI4MATH_LOOGLE_TIMEOUT", "10"))
        self.cache = cache if cache is not None else RetrievalCache.from_env()
        self._consecutive_failures = 0

    def available(self) -> bool:
        return self._consecutive_failures < 3

    def search(self, query: str, top_k: int = 10, *,
               goal_state: str = "") -> list[RetrievedPremise]:
        query = (query or "").strip()
        if not query:
            return []
        # 缓存键与 top_k 解耦: 同一查询只打一次 API, 存全量结果,
        # 不同 top_k 共享快照 (取前缀切片)。
        key = RetrievalCache.make_key(self.name, query, 0,
                                      extra=self.url)
        cached = self.cache.get(key)
        if cached is not None:
            return [RetrievedPremise(source=self.name, **d)
                    for d in cached][:top_k]
        if self.cache.mode == "ro":
            logger.warning("loogle: cache miss in ro mode for %r", query)
            return []

        full = f"{self.url}?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(full, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = json.loads(r.read().decode())
            self._consecutive_failures = 0
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError) as e:
            self._consecutive_failures += 1
            logger.warning("loogle unreachable: %s", e)
            return []

        if isinstance(raw, dict) and raw.get("error"):
            # Loogle 对非法模式返回 error 字段 — 这是查询语法问题,
            # 不是服务故障, 不计入熔断。
            logger.info("loogle query error: %s", raw["error"])
            return []
        hits = raw.get("hits", []) if isinstance(raw, dict) else []
        out = []
        for i, h in enumerate(hits[:max(top_k, 10)]):
            if not isinstance(h, dict) or not h.get("name"):
                continue
            out.append(RetrievedPremise(
                name=str(h["name"]),
                statement=str(h.get("type", "")),
                score=1.0 - i * 0.01,   # Loogle 不给分, 按序衰减
                source=self.name,
                module=str(h.get("module", "")),
                informal=str(h.get("doc", ""))[:300],
            ))
        self.cache.put(key, query, [
            {"name": p.name, "statement": p.statement, "score": p.score,
             "module": p.module, "informal": p.informal} for p in out])
        return out[:top_k]

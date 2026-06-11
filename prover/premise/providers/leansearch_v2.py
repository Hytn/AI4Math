"""prover/premise/providers/leansearch_v2.py — LeanSearch (v2) API provider

LeanSearch v2 (arXiv:2605.13137, 北大董彬团队) 是当前 Mathlib 语义检索
SOTA: standard mode 用 hierarchy-informalized 语料 + embedding-reranker,
nDCG@10 0.62; reasoning mode 在其上做 sketch-retrieve-reflect 循环面向
global premise retrieval。公开服务: https://leansearch.net。

本 provider 接的是它的 **standard mode HTTP API**。reasoning mode 的
sketch-retrieve-reflect 不在 provider 层做 —— 那是 agent loop 的职责
(ObservationPolicy / framing 层面的迭代检索), provider 只负责
"一条查询 → 一组 premise"。

工程要点:
  - 纯 stdlib urllib, 不新增依赖;
  - 所有请求过 RetrievalCache 快照 (评测可复现, 见 cache.py 模块注释);
  - endpoint 与请求格式可经环境变量覆盖 —— 外部 API 是移动目标,
    schema 变更时无需改代码:
        AI4MATH_LEANSEARCH_URL      (默认 https://leansearch.net/search)
        AI4MATH_LEANSEARCH_TIMEOUT  (默认 10 秒)
  - 响应解析做了多 schema 容错 (历史上 LeanSearch 返回格式有过变化);
    解析不出时返回空 + WARNING, 绝不抛异常到 agent loop。

⚠️ 离线环境 (无外网/防火墙) 下该 provider 自动降级为不可用 —
   MultiRetriever 会把它记入 degraded 列表, 由后续 provider 兜底。
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

_DEFAULT_URL = "https://leansearch.net/search"
_UA = "AI4Math-retriever/1.0 (+https://github.com/ai4math/ai4math)"


@register_provider
class LeanSearchV2Provider(RetrieverProvider):
    name = "leansearch_v2"

    def __init__(self, url: str = "", timeout: float = 0.0,
                 cache: RetrievalCache | None = None, **_):
        self.url = url or os.environ.get(
            "AI4MATH_LEANSEARCH_URL", _DEFAULT_URL)
        self.timeout = timeout or float(
            os.environ.get("AI4MATH_LEANSEARCH_TIMEOUT", "10"))
        self.cache = cache if cache is not None else RetrievalCache.from_env()
        self._consecutive_failures = 0

    def available(self) -> bool:
        # 连续失败 3 次后本进程内熔断, 避免每个题目都等超时。
        return self._consecutive_failures < 3

    # ─── 查询 ────────────────────────────────────────────────────

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
            return [self._from_dict(d) for d in cached][:top_k]
        if self.cache.mode == "ro":
            # 冻结快照评测: 未命中即降级, 不打 API。
            logger.warning(
                "leansearch_v2: cache miss in ro mode for query %r", query)
            return []

        raw = self._http_search(query, max(top_k, 10))
        if raw is None:
            return []
        premises = self._parse(raw)
        self.cache.put(key, query,
                       [self._to_dict(p) for p in premises])
        return premises[:top_k]

    # ─── HTTP ────────────────────────────────────────────────────

    def _http_search(self, query: str, top_k: int):
        """POST {url} — 兼容 LeanSearch 公开 API 的批量查询格式。

        请求体: [{"query": "...", "num_results": k}]
        (LeanSearch 接受批量列表; 我们每次只发一条。)
        若服务端拒绝列表格式 (400/422), 自动降级重试单对象格式
        {"query": "...", "num_results": k}。
        """
        for body in ([{"query": query, "num_results": top_k}],
                     {"query": query, "num_results": top_k}):
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                self.url, data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "User-Agent": _UA})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self._consecutive_failures = 0
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 405, 422):
                    continue  # 试下一种请求格式
                self._consecutive_failures += 1
                logger.warning("leansearch_v2 HTTP %s: %s", e.code, e.reason)
                return None
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as e:
                self._consecutive_failures += 1
                logger.warning("leansearch_v2 unreachable: %s", e)
                return None
        self._consecutive_failures += 1
        logger.warning("leansearch_v2: all request formats rejected by %s",
                       self.url)
        return None

    # ─── 解析 (多 schema 容错) ───────────────────────────────────

    def _parse(self, raw) -> list[RetrievedPremise]:
        """归一化已知的几种返回形态:

        A) [[{...hit...}, ...]]              — 批量查询, 外层 per-query
        B) [{...hit...}, ...]                — 单查询直接列表
        C) {"results"|"hits": [{...}, ...]}  — 包一层 dict
        hit 形态又分:
          {"result": {"name"/"formal_name", "statement"/"formal_type",
                      "module_name", "informal_name"/"informal_description",
                      "kind"}, "score"/"distance": ...}
          或扁平的同名字段。
        """
        hits = raw
        if isinstance(raw, dict):
            hits = raw.get("results") or raw.get("hits") or []
        if (isinstance(hits, list) and hits
                and isinstance(hits[0], list)):
            hits = hits[0]  # 批量外层取第一条查询
        if not isinstance(hits, list):
            logger.warning("leansearch_v2: unrecognised response shape %s",
                           type(raw).__name__)
            return []

        out: list[RetrievedPremise] = []
        for i, h in enumerate(hits):
            if not isinstance(h, dict):
                continue
            inner = h.get("result") if isinstance(h.get("result"), dict) else h
            name = (inner.get("formal_name") or inner.get("name")
                    or inner.get("full_name") or "")
            if not name:
                continue
            score = h.get("score", inner.get("score"))
            if score is None:
                dist = h.get("distance", inner.get("distance"))
                # 距离 → 相关性: 单调反转; 缺省按排名衰减。
                score = (1.0 / (1.0 + float(dist))
                         if dist is not None else 1.0 - i * 0.01)
            out.append(RetrievedPremise(
                name=str(name),
                statement=str(inner.get("formal_type")
                              or inner.get("statement") or ""),
                score=float(score),
                source=self.name,
                module=str(inner.get("module_name")
                           or inner.get("module") or ""),
                informal=str(inner.get("informal_description")
                             or inner.get("informal_name") or ""),
                kind=str(inner.get("kind") or ""),
            ))
        return out

    # ─── 缓存序列化 ─────────────────────────────────────────────

    @staticmethod
    def _to_dict(p: RetrievedPremise) -> dict:
        return {"name": p.name, "statement": p.statement, "score": p.score,
                "module": p.module, "informal": p.informal, "kind": p.kind}

    def _from_dict(self, d: dict) -> RetrievedPremise:
        return RetrievedPremise(
            name=d.get("name", ""), statement=d.get("statement", ""),
            score=float(d.get("score", 0.0)), source=self.name,
            module=d.get("module", ""), informal=d.get("informal", ""),
            kind=d.get("kind", ""))

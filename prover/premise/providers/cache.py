"""prover/premise/providers/cache.py — 在线检索的快照缓存

为什么必须有这个模块: LeanSearch / Loogle 等外部服务是**移动目标**
(索引随 Mathlib 滚动更新)。如果评测时直接打 API:

  1. 同一次评测重跑结果不可复现;
  2. 评测高并发会打爆/限流外部服务;
  3. dialog.json 里记录的检索结果与后续审计时的 API 返回对不上。

所以所有在线 provider 的查询都经过本缓存:
  - mode="rw"  (默认): 命中读缓存, 未命中打 API 并写入快照;
  - mode="ro": 只读重放 — 未命中返回 miss (provider 视为降级),
               用于"冻结快照"的可复现评测;
  - mode="off": 直通 (调试用)。

存储格式: JSONL append-only, 每行
  {"k": "<provider>\t<sha1(query|top_k|extra)>", "query": ..., "results": [...]}
进程内有 dict 索引; 文件锁省略 (append 单行写在 POSIX 下原子性足够,
并发评测建议每个 run 一个 cache 文件)。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "results/retrieval_cache.jsonl"


class RetrievalCache:
    def __init__(self, path: str = _DEFAULT_PATH, mode: str = "rw"):
        if mode not in ("rw", "ro", "off"):
            raise ValueError(f"cache mode must be rw/ro/off, got {mode!r}")
        self.path = path
        self.mode = mode
        self._lock = threading.Lock()
        self._index: dict[str, list[dict]] = {}
        if mode != "off":
            self._load()

    # ─── 构造 ────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, path: str = "", mode: str = "") -> "RetrievalCache":
        path = path or os.environ.get("AI4MATH_RETRIEVAL_CACHE", _DEFAULT_PATH)
        mode = mode or os.environ.get("AI4MATH_RETRIEVAL_CACHE_MODE", "rw")
        return cls(path=path, mode=mode)

    # ─── 键 ──────────────────────────────────────────────────────

    @staticmethod
    def make_key(provider: str, query: str, top_k: int,
                 extra: str = "") -> str:
        h = hashlib.sha1(
            f"{query}\x00{top_k}\x00{extra}".encode()).hexdigest()
        return f"{provider}\t{h}"

    # ─── 读写 ────────────────────────────────────────────────────

    def get(self, key: str):
        """命中 → list[dict]; 未命中 → None。off 模式恒 None。"""
        if self.mode == "off":
            return None
        with self._lock:
            return self._index.get(key)

    def put(self, key: str, query: str, results: list[dict]):
        if self.mode != "rw":
            return
        with self._lock:
            if key in self._index:
                return
            self._index[key] = results
            try:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(
                        {"k": key, "query": query, "results": results},
                        ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("RetrievalCache write failed (%s): %s",
                               self.path, e)

    def _load(self):
        if not os.path.exists(self.path):
            if self.mode == "ro":
                logger.warning(
                    "RetrievalCache mode=ro but snapshot %s does not exist — "
                    "all online retrievals will report as degraded.",
                    self.path)
            return
        n_bad = 0
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._index[rec["k"]] = rec["results"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        n_bad += 1
        except OSError as e:
            logger.warning("RetrievalCache load failed (%s): %s",
                           self.path, e)
        if n_bad:
            logger.warning("RetrievalCache: skipped %d malformed lines in %s",
                           n_bad, self.path)

    def __len__(self):
        return len(self._index)

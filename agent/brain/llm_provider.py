"""agent/brain/llm_provider.py — LLM Provider 抽象基类 + 缓存装饰器"""
from __future__ import annotations
import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    tool_calls: list = None
    raw_response: Optional[dict] = None
    cached: bool = False


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.7,
                 tools: list = None, max_tokens: int = 4096) -> LLMResponse: ...
    @property
    @abstractmethod
    def model_name(self) -> str: ...


class CachedProvider(LLMProvider):
    """Wraps any LLMProvider with an in-memory LRU cache.

    Cache key = hash(system + user + str(temperature) + str(tools)).
    Only deterministic calls (temperature=0) are cached by default,
    or all calls if ``cache_all=True``.

    Thread-safe: all cache operations are protected by a lock.

    Usage::

        base = ClaudeProvider(...)
        cached = CachedProvider(base, maxsize=512)
        resp = cached.generate(system="...", user="...", temperature=0)
        # Second call with same args returns cached result instantly
    """

    def __init__(self, provider: LLMProvider, maxsize: int = 512,
                 cache_all: bool = False):
        self._provider = provider
        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._maxsize = maxsize
        self._cache_all = cache_all
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def generate(self, system: str = "", user: str = "",
                 temperature: float = 0.7,
                 tools: list = None, max_tokens: int = 4096) -> LLMResponse:
        # Only cache deterministic calls unless cache_all
        cacheable = self._cache_all or temperature == 0
        if cacheable:
            key = self._make_key(system, user, temperature, max_tokens, tools)
            with self._lock:
                if key in self._cache:
                    self.hits += 1
                    self._cache.move_to_end(key)
                    resp = self._cache[key]
                    # Return a copy with cached=True
                    return LLMResponse(
                        content=resp.content, model=resp.model,
                        tokens_in=0, tokens_out=0, latency_ms=0,
                        tool_calls=resp.tool_calls, cached=True)

        self.misses += 1
        resp = self._provider.generate(
            system=system, user=user, temperature=temperature,
            tools=tools, max_tokens=max_tokens)

        if cacheable:
            key = self._make_key(system, user, temperature, max_tokens, tools)
            with self._lock:
                self._cache[key] = resp
                self._cache.move_to_end(key)
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

        return resp

    def cache_stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0,
                "size": len(self._cache),
            }

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @staticmethod
    def _make_key(system: str, user: str, temperature: float,
                  max_tokens: int, tools: list = None) -> str:
        # Include tools in the cache key to avoid cross-contamination
        tools_str = str(sorted(str(t) for t in tools)) if tools else ""
        raw = f"{system}|||{user}|||{temperature}|||{max_tokens}|||{tools_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

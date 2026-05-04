"""agent/brain/async_llm_provider.py — 异步 LLM Provider

与同步 LLMProvider 共享 LLMResponse 数据类型,
但 generate() 改为 async, 底层使用 anthropic AsyncAnthropic。

用法::

    from common.constants import DEFAULT_CLAUDE_MODEL
    provider = AsyncClaudeProvider(model=DEFAULT_CLAUDE_MODEL)
    resp = await provider.generate(system="...", user="...")
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from common.constants import DEFAULT_CLAUDE_MODEL, ANTHROPIC_API_KEY_ENV

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


@dataclass
class LLMResponse:
    """LLM call result. Shared across sync/async paths.

    Migrated to async_llm_provider in v9 (sync llm_provider.py deleted).
    """
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    tool_calls: list = None
    raw_response: Optional[dict] = None
    cached: bool = False
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens"


class AsyncLLMProvider(ABC):
    """异步 LLM Provider 抽象基类"""

    @abstractmethod
    async def generate(self, system: str, user: str,
                       temperature: float = 0.7,
                       tools: list = None,
                       max_tokens: int = 4096) -> LLMResponse: ...

    async def chat(self, system: str, messages: list[dict],
                   temperature: float = 0.7, tools: list = None,
                   max_tokens: int = 4096) -> LLMResponse:
        """Multi-turn chat with full messages array.

        Default falls back to generate() with last user message.
        Providers should override for proper multi-turn support.
        """
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user = content
                elif isinstance(content, list):
                    last_user = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
                break
        return await self.generate(system, last_user, temperature, tools, max_tokens)

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class AsyncClaudeProvider(AsyncLLMProvider):
    """Claude 异步 Provider — 使用 anthropic.AsyncAnthropic

    关键差异:
      - 同步版: client.messages.create() → 阻塞线程等待 HTTP 响应
      - 异步版: await client.messages.create() → 释放事件循环, 可并行
    """

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL,
                 api_key: str = ""):
        self._model = model
        self._api_key = api_key or os.environ.get(ANTHROPIC_API_KEY_ENV, "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system: str = "", user: str = "",
                       temperature: float = 0.7,
                       tools: list = None,
                       max_tokens: int = 4096) -> LLMResponse:
        return await self.chat(system, [{"role": "user", "content": user}],
                               temperature, tools, max_tokens)

    async def chat(self, system: str = "", messages: list[dict] = None,
                   temperature: float = 0.7, tools: list = None,
                   max_tokens: int = 4096) -> LLMResponse:
        """Multi-turn async chat supporting tool_use conversations."""
        client = self._get_client()
        kwargs = dict(
            model=self._model, max_tokens=max_tokens,
            temperature=temperature, system=system,
            messages=messages or [])
        if tools:
            kwargs["tools"] = tools

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                start = time.time()
                response = await client.messages.create(**kwargs)
                latency = int((time.time() - start) * 1000)

                content = ""
                tool_calls = []
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "name": block.name,
                            "input": block.input,
                            "id": block.id})

                return LLMResponse(
                    content=content, model=self._model,
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    latency_ms=latency, tool_calls=tool_calls,
                    stop_reason=response.stop_reason or "end_turn")

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    backoff = min(
                        _INITIAL_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                    jitter = random.uniform(0, backoff * 0.3)
                    wait = backoff + jitter
                    logger.warning(
                        f"Async Claude API failed (attempt {attempt + 1}/"
                        f"{_MAX_RETRIES + 1}): {e}. Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"Async Claude API failed after "
                        f"{_MAX_RETRIES + 1} attempts: {e}")

        raise last_error

    async def close(self):
        """关闭底层 HTTP 连接池"""
        if self._client:
            await self._client.close()
            self._client = None


class AsyncMockProvider(AsyncLLMProvider):
    """测试用异步 Mock Provider"""

    @property
    def model_name(self) -> str:
        return "async-mock"

    async def generate(self, system="", user="", temperature=0.7,
                       tools=None, max_tokens=4096) -> LLMResponse:
        return await self.chat(system, [{"role": "user", "content": user}],
                               temperature, tools, max_tokens)

    async def chat(self, system="", messages=None, temperature=0.7,
                   tools=None, max_tokens=4096) -> LLMResponse:
        await asyncio.sleep(0.01)
        return LLMResponse(
            content="```lean\n:= by\n  sorry\n```",
            model="async-mock", tokens_in=100, tokens_out=20,
            latency_ms=10, stop_reason="end_turn")


class AsyncCachedProvider(AsyncLLMProvider):
    """异步缓存包装器 — 包装任意 AsyncLLMProvider.

    v12: now also overrides ``chat()``. ``AgentLoop`` calls ``chat``
    preferentially (see ``agent/runtime/agent_loop.py``), so prior to
    v12 only the rarer ``generate()`` path was cached and the headline
    multi-turn loop bypassed the cache entirely.
    """

    def __init__(self, provider: AsyncLLMProvider, maxsize: int = 512,
                 cache_all: bool = False):
        self._provider = provider
        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._maxsize = maxsize
        self._cache_all = cache_all
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    async def generate(self, system: str = "", user: str = "",
                       temperature: float = 0.7,
                       tools: list = None,
                       max_tokens: int = 4096) -> LLMResponse:
        cacheable = self._cache_all or temperature == 0
        if cacheable:
            key = self._make_key(system, user, temperature, max_tokens, tools)
            async with self._lock:
                if key in self._cache:
                    self.hits += 1
                    self._cache.move_to_end(key)
                    resp = self._cache[key]
                    return LLMResponse(
                        content=resp.content, model=resp.model,
                        tokens_in=0, tokens_out=0, latency_ms=0,
                        tool_calls=resp.tool_calls, cached=True)

        self.misses += 1
        resp = await self._provider.generate(
            system=system, user=user, temperature=temperature,
            tools=tools, max_tokens=max_tokens)

        if cacheable:
            key = self._make_key(system, user, temperature, max_tokens, tools)
            async with self._lock:
                self._cache[key] = resp
                self._cache.move_to_end(key)
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

        return resp

    async def chat(self, system: str = "", messages: list = None,
                   temperature: float = 0.7, tools: list = None,
                   max_tokens: int = 4096) -> LLMResponse:
        """Cached chat: keys on (system, full messages, tools, T, max_tokens).

        Multi-turn proof loops re-run the same prefix repeatedly while
        searching, so even non-zero temperature can benefit when
        ``cache_all=True``. The default policy still only caches
        deterministic (T=0) calls to avoid surprising users.
        """
        import json as _json
        cacheable = self._cache_all or temperature == 0
        if cacheable:
            try:
                msgs_str = _json.dumps(messages or [], sort_keys=True,
                                        default=str)
            except Exception:
                msgs_str = repr(messages)
            key = self._make_key(system, msgs_str, temperature,
                                  max_tokens, tools)
            async with self._lock:
                if key in self._cache:
                    self.hits += 1
                    self._cache.move_to_end(key)
                    resp = self._cache[key]
                    return LLMResponse(
                        content=resp.content, model=resp.model,
                        tokens_in=0, tokens_out=0, latency_ms=0,
                        tool_calls=resp.tool_calls,
                        stop_reason=resp.stop_reason, cached=True)

        self.misses += 1
        # Delegate to underlying provider's chat() if present, else its
        # default (which falls back to generate() with last user msg).
        resp = await self._provider.chat(
            system=system, messages=messages or [],
            temperature=temperature, tools=tools, max_tokens=max_tokens)

        if cacheable:
            async with self._lock:
                self._cache[key] = resp
                self._cache.move_to_end(key)
                if len(self._cache) > self._maxsize:
                    self._cache.popitem(last=False)

        return resp

    def cache_stats(self) -> dict:
        """Hit/miss summary; useful for the CLI to print at end of run."""
        total = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "size": len(self._cache),
            "hit_rate": (self.hits / total) if total else 0.0,
        }

    @staticmethod
    def _make_key(system, user, temperature, max_tokens, tools=None) -> str:
        tools_str = str(sorted(str(t) for t in tools)) if tools else ""
        raw = f"{system}|||{user}|||{temperature}|||{max_tokens}|||{tools_str}"
        return hashlib.sha256(raw.encode()).hexdigest()


def create_async_provider(config: dict) -> AsyncLLMProvider:
    """工厂: 从配置创建异步 Provider"""
    p = config.get("provider", "anthropic")
    if p == "anthropic":
        return AsyncClaudeProvider(
            model=config.get("model", DEFAULT_CLAUDE_MODEL),
            api_key=config.get("api_key", ""))
    elif p == "mock":
        return AsyncMockProvider()
    else:
        raise ValueError(f"Unknown provider: {p}")

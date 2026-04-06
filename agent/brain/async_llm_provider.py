"""agent/brain/async_llm_provider.py — 异步 LLM Provider

与同步 LLMProvider 共享 LLMResponse 数据类型,
但 generate() 改为 async, 底层使用 anthropic AsyncAnthropic。

用法::

    provider = AsyncClaudeProvider(model="claude-sonnet-4-20250514")
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
from typing import Optional

from agent.brain.llm_provider import LLMResponse

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


class AsyncLLMProvider(ABC):
    """异步 LLM Provider 抽象基类"""

    @abstractmethod
    async def generate(self, system: str, user: str,
                       temperature: float = 0.7,
                       tools: list = None,
                       max_tokens: int = 4096) -> LLMResponse: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


class AsyncClaudeProvider(AsyncLLMProvider):
    """Claude 异步 Provider — 使用 anthropic.AsyncAnthropic

    关键差异:
      - 同步版: client.messages.create() → 阻塞线程等待 HTTP 响应
      - 异步版: await client.messages.create() → 释放事件循环, 可并行
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514",
                 api_key: str = ""):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
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
        client = self._get_client()
        kwargs = dict(
            model=self._model, max_tokens=max_tokens,
            temperature=temperature, system=system,
            messages=[{"role": "user", "content": user}])
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
                    latency_ms=latency, tool_calls=tool_calls)

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
        await asyncio.sleep(0.01)  # 模拟网络延迟
        return LLMResponse(
            content="```lean\n:= by\n  sorry\n```",
            model="async-mock", tokens_in=100, tokens_out=20,
            latency_ms=10)


class AsyncCachedProvider(AsyncLLMProvider):
    """异步缓存包装器 — 包装任意 AsyncLLMProvider"""

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
            model=config.get("model", "claude-sonnet-4-20250514"),
            api_key=config.get("api_key", ""))
    elif p == "mock":
        return AsyncMockProvider()
    else:
        raise ValueError(f"Unknown provider: {p}")

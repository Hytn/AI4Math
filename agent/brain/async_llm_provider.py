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
    """Mock LLM provider — 用于无网络的单元测试和管线冒烟。

    重要:这个 provider 的输出**不是真实证明**,只是一组常用 tactic 的脚本
    回放(omega / simp / rfl / ring / decide)。它能让 ``MockTransport`` 跑
    通端到端管线,但**不能用于评测**。任何 dialog.json 用 mock 跑出来的
    ``meta.provider == "mock"`` 字样应被评测脚本视为非有效证明。

    可以通过 ``MockResponseScript`` 注入定制响应(用于复现具体场景)。
    """

    def __init__(self, scripted_responses: Optional[list[str]] = None):
        # 优先吐脚本里的响应;吐完了用启发式
        self._scripted = list(scripted_responses or [])

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
        messages = messages or []

        # 1. 脚本响应优先 (LLMResponse 形式 / 字符串都接受)
        if self._scripted:
            scr = self._scripted.pop(0)
            if isinstance(scr, LLMResponse):
                return scr
            return LLMResponse(
                content=str(scr),
                model="async-mock", tokens_in=100, tokens_out=20,
                latency_ms=10, stop_reason="end_turn")

        user_text = self._last_user_text(messages)
        proof = self._heuristic_proof(user_text)

        # 2. tool-call 路径
        # 优先级:lean_verify (whole-proof tool) > tactic_apply (step-level)
        tool_names = {self._tool_name(t) for t in (tools or [])}
        already_got_tool_result = any(
            (m.get("role") == "tool")
            or (m.get("role") == "user"
                and isinstance(m.get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in m["content"]))
            for m in messages)

        # 已经收到过 tool_result 时直接终止 — 无论之前调的是哪个工具,
        # 第二轮都给 raw proof,fast-path 接住,success=True。
        if already_got_tool_result:
            return LLMResponse(
                content=proof,
                model="async-mock", tokens_in=100, tokens_out=20,
                latency_ms=10, stop_reason="end_turn")

        if "lean_verify" in tool_names:
            theorem = self._extract_theorem(user_text) \
                or "theorem mock_t : True"
            proof_body = (proof.replace("```lean\n", "")
                              .replace("\n```", "").strip())
            full_code = f"{theorem} {proof_body}" if proof_body.startswith(":=") \
                else f"{theorem} := by {proof_body}"
            tool_call = {
                "id": f"call_mock_{random.randint(0, 999999)}",
                "name": "lean_verify",
                "input": {"code": full_code},
            }
            return LLMResponse(
                content="",
                model="async-mock", tokens_in=100, tokens_out=20,
                latency_ms=10, stop_reason="tool_use",
                tool_calls=[tool_call])

        if "tactic_apply" in tool_names:
            # step-level:挑一个启发式 tactic 让 driver 至少跑一轮 expansion
            tactic = self._heuristic_tactic(user_text)
            tool_call = {
                "id": f"call_mock_{random.randint(0, 999999)}",
                "name": "tactic_apply",
                "input": {"tactic": tactic},
            }
            return LLMResponse(
                content="",
                model="async-mock", tokens_in=100, tokens_out=20,
                latency_ms=10, stop_reason="tool_use",
                tool_calls=[tool_call])

        if "lemma_by_lemma" in tool_names:
            # LooKeng profile:发一次 is_final=True 的 final-lemma 调用,
            # 直接把整道定理作为最终引理证明出来
            theorem = self._extract_theorem(user_text) \
                or "theorem mock_t : True"
            tactic = self._heuristic_tactic(user_text)
            tool_call = {
                "id": f"call_mock_{random.randint(0, 999999)}",
                "name": "lemma_by_lemma",
                "input": {
                    "name": "final_lemma",
                    "statement": theorem,
                    "proof": tactic,
                    "is_final": True,
                },
            }
            return LLMResponse(
                content="",
                model="async-mock", tokens_in=100, tokens_out=20,
                latency_ms=10, stop_reason="tool_use",
                tool_calls=[tool_call])

        return LLMResponse(
            content=proof,
            model="async-mock", tokens_in=100, tokens_out=20,
            latency_ms=10, stop_reason="end_turn")

    @staticmethod
    def _tool_name(t) -> str:
        if isinstance(t, dict):
            return t.get("name") or t.get("function", {}).get("name", "")
        return getattr(t, "name", "") or getattr(
            getattr(t, "function", None), "name", "")

    @staticmethod
    def _extract_theorem(text: str) -> str:
        """从 user 文本里捞第一段 ```lean ... ``` 中的 theorem 头。"""
        import re
        m = re.search(r"```lean\s*\n(.+?)\n```", text, re.DOTALL)
        if not m:
            return ""
        for line in m.group(1).splitlines():
            stripped = line.strip()
            if stripped.startswith(("theorem ", "lemma ", "example ")):
                return stripped.rstrip(":=").rstrip(":").strip()
        return ""

    @staticmethod
    def _last_user_text(messages: list) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return " ".join(
                        b.get("text", "") for b in c
                        if isinstance(b, dict) and b.get("type") == "text")
        return ""

    @staticmethod
    def _heuristic_proof(user_text: str) -> str:
        """从题面挑一个看起来最可能的 tactic。无网络环境冒烟用。"""
        t = user_text.lower()
        if "comm" in t or "+" in t or "* " in t:
            tactic = "omega"
        elif "iff" in t or "↔" in t:
            tactic = "tauto"
        elif "=" in t and ("nat" in t or "int" in t or "ℕ" in t or "ℤ" in t):
            tactic = "omega"
        elif "le" in t or "≤" in t or "<" in t:
            tactic = "linarith"
        elif "true" in t or "trivial" in t:
            tactic = "trivial"
        else:
            tactic = "simp"
        return f"```lean\n:= by\n  {tactic}\n```"

    @classmethod
    def _heuristic_tactic(cls, user_text: str) -> str:
        """tactic_apply 用的:从启发式 proof 里挑出 tactic 单词。"""
        body = cls._heuristic_proof(user_text)
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("`", ":=")):
                return stripped
        return "simp"

# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────

class AsyncOpenAIProvider(AsyncLLMProvider):
    """OpenAI Chat Completions-compatible async provider.

    Designed to be the single Python class that talks to anything
    speaking OpenAI's API shape — DeepSeek API, vLLM, sglang, ollama,
    Together / Anyscale / Fireworks / Groq, plus OpenAI itself.

    Why this matters for AI4Math:

      1. ``--profile whole_proof`` claims "DeepSeek-Prover style" but
         until v15 only Anthropic was wired in. Now you can A/B Claude
         vs DeepSeek-Prover-V2 vs Kimina-Prover on the same harness.

      2. The RL flywheel in ``sampler/`` is economically infeasible
         when every rollout calls Anthropic. With vLLM via this
         provider, a single H100 can serve thousands of rollouts/min.

      3. Lets evaluators reproduce numbers without an Anthropic
         account — point ``--api-base`` at any OpenAI-compatible server.

    Tool-use translation:
      The agent loop speaks Anthropic's tool format internally
      (content blocks with ``type=tool_use``). This provider:
        * sends tools as OpenAI ``functions``-style schema
        * receives ``tool_calls`` from OpenAI back
        * normalises the response to the same ``LLMResponse(tool_calls=[
          {"name", "input", "id"}, ...])`` shape AsyncClaudeProvider returns
      so AgentLoop sees identical structure regardless of backend.

    Usage::

        # DeepSeek API
        provider = AsyncOpenAIProvider(
            model="deepseek-chat",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            api_base="https://api.deepseek.com/v1",
        )

        # Local vLLM
        provider = AsyncOpenAIProvider(
            model="DeepSeek-Prover-
            api_key="EMPTY",
            api_base="http://localhost:8000/v1",
        )
    """

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: str = "",
                 api_base: str = "",
                 timeout_s: float = 120.0):
        self._model = model
        # Allow empty api_key for local servers (vLLM/sglang/ollama).
        # The OpenAI client requires *some* string, so default to
        # "EMPTY" which is the vLLM convention.
        self._api_key = api_key or os.environ.get(
            "OPENAI_API_KEY", "") or "EMPTY"
        self._api_base = api_base or os.environ.get("OPENAI_API_BASE", "")
        self._timeout_s = timeout_s
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise RuntimeError(
                    "AsyncOpenAIProvider requires the 'openai' package. "
                    "Install with: pip install openai>=1.40.0"
                ) from e
            kwargs = {"api_key": self._api_key, "timeout": self._timeout_s}
            if self._api_base:
                kwargs["base_url"] = self._api_base
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @staticmethod
    def _claude_tools_to_openai(tools: list) -> list:
        """Convert Anthropic tool schema to OpenAI function-calling schema.

        Anthropic format:  {"name", "description", "input_schema"}
        OpenAI   format:   {"type":"function",
                            "function":{"name","description","parameters"}}
        """
        out = []
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema",
                                         {"type": "object", "properties": {}}),
                },
            })
        return out

    @staticmethod
    def _claude_messages_to_openai(messages: list) -> list:
        """Translate Anthropic content-block messages to OpenAI shape.

        Anthropic uses structured ``content`` lists with text blocks and
        tool_use / tool_result blocks. OpenAI uses string ``content`` plus
        a separate ``tool_calls`` array on assistant messages and
        ``role="tool"`` messages for tool results.
        """
        import json as _json
        out = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            # content is a list of blocks
            text_parts = []
            tool_calls = []
            tool_results = []  # list of {tool_call_id, content}
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": _json.dumps(
                                block.get("input", {}),
                                ensure_ascii=False),
                        },
                    })
                elif btype == "tool_result":
                    tc_id = block.get("tool_use_id", "")
                    inner = block.get("content", "")
                    if isinstance(inner, list):
                        # extract text from inner blocks
                        inner = "".join(
                            ib.get("text", "")
                            for ib in inner
                            if isinstance(ib, dict) and ib.get("type") == "text"
                        )
                    tool_results.append({
                        "tool_call_id": tc_id, "content": str(inner)})

            joined_text = "\n".join(p for p in text_parts if p)

            if role == "assistant" and tool_calls:
                msg = {"role": "assistant", "content": joined_text or None,
                       "tool_calls": tool_calls}
                out.append(msg)
            elif tool_results:
                # tool_result blocks become separate role=tool messages
                for tr in tool_results:
                    out.append({"role": "tool", **tr})
                if joined_text:
                    out.append({"role": role, "content": joined_text})
            else:
                out.append({"role": role, "content": joined_text})
        return out

    async def generate(self, system: str = "", user: str = "",
                       temperature: float = 0.7,
                       tools: list = None,
                       max_tokens: int = 4096) -> LLMResponse:
        return await self.chat(
            system, [{"role": "user", "content": user}],
            temperature, tools, max_tokens)

    async def chat(self, system: str = "", messages: list = None,
                   temperature: float = 0.7, tools: list = None,
                   max_tokens: int = 4096) -> LLMResponse:
        client = self._get_client()
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(self._claude_messages_to_openai(messages or []))

        kwargs = dict(
            model=self._model, max_tokens=max_tokens,
            temperature=temperature, messages=oai_messages)
        if tools:
            kwargs["tools"] = self._claude_tools_to_openai(tools)

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                start = time.time()
                response = await client.chat.completions.create(**kwargs)
                latency = int((time.time() - start) * 1000)
                choice = response.choices[0]
                msg = choice.message

                content = msg.content or ""
                tool_calls_out = []
                for tc in (getattr(msg, "tool_calls", None) or []):
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    import json as _json
                    try:
                        parsed_args = _json.loads(fn.arguments or "{}")
                    except Exception:
                        parsed_args = {}
                    tool_calls_out.append({
                        "name": fn.name,
                        "input": parsed_args,
                        "id": getattr(tc, "id", "") or "",
                    })

                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "function_call": "tool_use",
                }
                stop_reason = stop_reason_map.get(
                    choice.finish_reason or "stop", "end_turn")

                usage = getattr(response, "usage", None)
                tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
                tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

                return LLMResponse(
                    content=content, model=self._model,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    latency_ms=latency, tool_calls=tool_calls_out,
                    stop_reason=stop_reason)

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    backoff = min(
                        _INITIAL_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                    jitter = random.uniform(0, backoff * 0.3)
                    wait = backoff + jitter
                    logger.warning(
                        f"AsyncOpenAIProvider call failed (attempt "
                        f"{attempt + 1}/{_MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"AsyncOpenAIProvider failed after "
                        f"{_MAX_RETRIES + 1} attempts: {e}")

        raise last_error

    async def close(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

class AsyncCachedProvider(AsyncLLMProvider):
    """异步缓存包装器 — 包装任意 AsyncLLMProvider.

    
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
    """工厂: 从配置创建异步 Provider.

    
    api_base / model / env-var defaults on AsyncOpenAIProvider; the
    explicit ``api_base`` / ``model`` / ``api_key`` in ``config``
    always wins.

    Recognised provider names::

        "anthropic"     →  AsyncClaudeProvider           (Claude API)
        "mock"          →  AsyncMockProvider             (no network)
        "openai"        →  AsyncOpenAIProvider           (OpenAI API)
        "deepseek"      →  AsyncOpenAIProvider           (api.deepseek.com)
        "vllm"          →  AsyncOpenAIProvider           (localhost:8000)
        "sglang"        →  AsyncOpenAIProvider           (localhost:30000)
        "ollama"        →  AsyncOpenAIProvider           (localhost:11434)
        "openai_compat" →  AsyncOpenAIProvider           (generic; expects
                                                          api_base in config)
    """
    p = (config.get("provider") or "anthropic").lower()
    api_key = config.get("api_key", "")
    api_base = config.get("api_base", "")
    model = config.get("model", "")

    if p == "anthropic":
        return AsyncClaudeProvider(
            model=model or DEFAULT_CLAUDE_MODEL,
            api_key=api_key)
    if p == "mock":
        return AsyncMockProvider()

    # OpenAI-compatible family — pick defaults per alias, then let
    # explicit config override.
    _OAI_ALIASES = {
        "openai":        ("",                                "gpt-4o-mini",
                          "OPENAI_API_KEY"),
        "deepseek":      ("https://api.deepseek.com/v1",     "deepseek-chat",
                          "DEEPSEEK_API_KEY"),
        "vllm":          ("http://localhost:8000/v1",        "",
                          ""),
        "sglang":        ("http://localhost:30000/v1",       "",
                          ""),
        "ollama":        ("http://localhost:11434/v1",       "",
                          ""),
        "openai_compat": ("",                                "",
                          ""),
    }
    if p in _OAI_ALIASES:
        default_base, default_model, env_var = _OAI_ALIASES[p]
        if not api_base:
            api_base = default_base
        if not model:
            model = default_model
        if not api_key and env_var:
            api_key = os.environ.get(env_var, "")
        if not model:
            raise ValueError(
                f"provider={p!r}: 'model' must be specified "
                f"(local servers don't have a default).")
        return AsyncOpenAIProvider(
            model=model, api_key=api_key, api_base=api_base)

    raise ValueError(
        f"Unknown provider: {p!r}. "
        f"Try one of: anthropic, mock, openai, deepseek, vllm, "
        f"sglang, ollama, openai_compat")

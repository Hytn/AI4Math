"""sampler/policy_adapter.py — Policy function adapters

The sampler talks to "the policy" exclusively through the
``PolicyFn = Callable[[obs], Awaitable[(text, token_ids, logprobs)]]``
type. v7 left it as the user's job to construct that callable. v7.1
ships the three obvious adapters so the most common cases are one
line of code.

Adapters
--------

``MockPolicy``
    Deterministic, no network. Returns a tactic from a configurable
    list. The default list ``["intro h", "exact h", "simp", "ring"]``
    pairs with the mock ProofEnv used in unit tests.

``OpenAIPolicy``
    OpenAI / vLLM / SGLang compatible. Calls an OpenAI-format
    ``/v1/chat/completions`` endpoint. Returns text + tokenised ids
    + per-token logprobs (when the server supplies them). Works with:
      * OpenAI Platform
      * Anthropic via OpenAI compatibility shim
      * Local vLLM:    ``vllm serve <model> --port 8001``
      * Local SGLang:  ``python -m sglang.launch_server --model-path <m>``

``CallablePolicy``
    Trivial wrapper around any sync ``f(obs) -> str`` for quick demos.

Tokeniser handling
------------------

Most production policies want token-level data so the RL trainer can
do per-token advantage assignment. ``OpenAIPolicy`` accepts an
optional ``tokenizer`` (the HuggingFace ``AutoTokenizer`` shape) and
calls it on the response text when the server didn't return token IDs
inline. If you don't pass a tokenizer, you get the text + empty
token_ids — fine for offline SFT collection, not enough for PPO.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# MockPolicy — deterministic, no LLM needed (for tests / demos)
# ═══════════════════════════════════════════════════════════════════════

class MockPolicy:
    """Deterministic policy that cycles through a fixed tactic list.

    Useful for:
      * Unit tests that need a non-trivial policy without an LLM.
      * Smoke-testing the full RL pipeline (sampler → trajectory →
        SFT JSONL → trainer subprocess) without paying API costs.
      * The ``scripts/rl_demo.py`` end-to-end demo.

    Args:
        tactics: list of tactic strings to cycle through. The default
            pairs with the mock ProofEnv used 's unit tests
            (which treats ``"exact h"`` as an instant win).
        seed: shuffle seed for ``shuffle=True``. Default 0 = deterministic.
        shuffle: if True, the cycle is shuffled per-call (so K
            candidates from the same observation aren't identical).
            Default False (strict cycle = trivially reproducible).
        token_ids_for: optional mapping ``tactic -> list[int]``. Lets
            tests pin specific token sequences for verifying token-level
            data flow. When unset, ``token_ids = []``.
    """

    def __init__(self,
                 tactics: list[str] = None,
                 seed: int = 0,
                 shuffle: bool = False,
                 token_ids_for: dict[str, list[int]] = None):
        self._tactics = list(tactics) if tactics else [
            "intro h", "exact h", "simp", "ring",
        ]
        self._idx = 0
        self._rng = random.Random(seed)
        self._shuffle = shuffle
        self._token_ids = token_ids_for or {}

    async def __call__(self, observation: str
                          ) -> tuple[str, list[int], list[float]]:
        if self._shuffle:
            tactic = self._rng.choice(self._tactics)
        else:
            tactic = self._tactics[self._idx % len(self._tactics)]
            self._idx += 1
        token_ids = self._token_ids.get(tactic, [])
        log_probs = [-1.0] * len(token_ids)
        return tactic, token_ids, log_probs

# ═══════════════════════════════════════════════════════════════════════
# CallablePolicy — wrap any sync f(obs) -> str
# ═══════════════════════════════════════════════════════════════════════

class CallablePolicy:
    """Wrap any callable ``f(obs) -> str`` as a policy_fn.

    Helpful when you have a heuristic prover and just want to test the
    RL pipeline on it without writing the policy_fn protocol by hand.
    """

    def __init__(self, fn: Callable[[str], str]):
        self._fn = fn

    async def __call__(self, observation: str
                          ) -> tuple[str, list[int], list[float]]:
        result = self._fn(observation)
        if asyncio.iscoroutine(result):
            text = await result
        else:
            text = result
        return text, [], []

# ═══════════════════════════════════════════════════════════════════════
# OpenAIPolicy — REST adapter for OpenAI / vLLM / SGLang servers
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in Lean 4 formal mathematics. "
    "Given a goal, respond with a single Lean 4 tactic that makes "
    "progress toward the proof. Output only the tactic, no explanation, "
    "no Markdown fences. Do not use `sorry`."
)

class OpenAIPolicy:
    """Policy adapter for OpenAI-compatible chat-completions endpoints.

    This is what makes a real RL roll-out runnable: paired with a
    locally-served vLLM / SGLang process, it gives the sampler a real
    policy without depending on verl/slime's server_manager.

    Token IDs and per-token logprobs are extracted from the response
    when the server returns them (vLLM and SGLang both support
    ``logprobs=True``). When they're missing and a HuggingFace
    ``AutoTokenizer`` is supplied via ``tokenizer=``, we tokenise the
    text locally; logprobs in that case are not available — set
    ``logprobs_for_unsourced=True`` to fill with placeholder zeros so
    PPO has *some* number to multiply against (advantages will still
    be wrong-but-bounded; this is the right behaviour when you only
    plan to do offline SFT and want token data for accounting).

    Args:
        base_url:        endpoint root, e.g. ``"http://localhost:8001/v1"``
        model:           model name (forwarded as ``"model"`` field)
        api_key:         bearer token (forwarded as ``Authorization``)
        tokenizer:       optional HuggingFace AutoTokenizer
        system_prompt:   prompt template prepended to each call
        temperature:     sampling temperature
        max_tokens:      cap on response length
        timeout_s:       request timeout
        logprobs_for_unsourced: see above
    """

    def __init__(self,
                 base_url: str = None,
                 model: str = "gpt-4o-mini",
                 api_key: str = None,
                 tokenizer: Any = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 temperature: float = 0.9,
                 max_tokens: int = 256,
                 timeout_s: float = 60.0,
                 logprobs_for_unsourced: bool = False):
        self.base_url = (base_url
                         or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model
        self.api_key = (api_key
                        or os.environ.get("OPENAI_API_KEY")
                        or "EMPTY")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.logprobs_for_unsourced = logprobs_for_unsourced
        self._session: Any = None  # lazy aiohttp.ClientSession

    async def __call__(self, observation: str
                          ) -> tuple[str, list[int], list[float]]:
        text = ""
        token_ids: list[int] = []
        log_probs: list[float] = []

        try:
            text, token_ids, log_probs = await self._chat_completion(
                observation)
        except Exception as e:
            logger.warning("OpenAIPolicy request failed: %r", e)
            return "", [], []

        # Extract Lean tactic from possibly-markdown-fenced response.
        text = self._unwrap(text)

        # If the server didn't supply token IDs but we have a tokenizer,
        # tokenise locally so the trainer has *something*.
        if not token_ids and self.tokenizer is not None:
            try:
                token_ids = self.tokenizer.encode(text)
                if self.logprobs_for_unsourced:
                    log_probs = [0.0] * len(token_ids)
                else:
                    log_probs = []
            except Exception as e:
                logger.debug("local tokenise failed: %r", e)

        return text, token_ids, log_probs

    async def _chat_completion(self, observation: str) -> tuple[
            str, list[int], list[float]]:
        """Lazy aiohttp post to /v1/chat/completions.

        We import aiohttp lazily so users without a real backend can
        still import OpenAIPolicy (e.g. for type checking / config).
        """
        try:
            import aiohttp  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OpenAIPolicy requires aiohttp. "
                "`pip install aiohttp` or use MockPolicy / CallablePolicy."
            ) from e

        if self._session is None:
            self._session = aiohttp.ClientSession()

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": observation},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "logprobs": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self._session.post(
                f"{self.base_url}/chat/completions",
                json=body, headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
        ) as resp:
            data = await resp.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {}) or {}
        text = message.get("content", "") or ""

        # Token IDs / logprobs (vLLM and SGLang return them inline).
        token_ids: list[int] = []
        log_probs: list[float] = []
        lp_block = choice.get("logprobs") or {}
        # Layout 1 (OpenAI / SGLang): logprobs.content = [{token, logprob, ...}, ...]
        for item in lp_block.get("content", []) or []:
            if "logprob" in item:
                log_probs.append(item["logprob"])
            tid = item.get("token_id") or item.get("id")
            if tid is not None:
                token_ids.append(int(tid))
        # Layout 2 (some vLLM versions): logprobs.tokens / .token_logprobs
        if not log_probs and "token_logprobs" in lp_block:
            log_probs = [
                lp for lp in lp_block.get("token_logprobs", [])
                if lp is not None
            ]

        return text, token_ids, log_probs

    @staticmethod
    def _unwrap(text: str) -> str:
        """Pull a Lean tactic out of a Markdown-fenced or padded response.

        We accept the most common shapes the model produces:
          * raw tactic   →  used as-is
          * ```lean\\nT\\n```  → T
          * ```\\nT\\n``` → T
        """
        text = text.strip()
        if "```lean" in text:
            try:
                start = text.index("```lean") + len("```lean")
                end = text.index("```", start)
                return text[start:end].strip()
            except ValueError:
                pass
        if text.startswith("```"):
            try:
                end = text.index("```", 3)
                return text[3:end].strip()
            except ValueError:
                pass
        return text

    async def close(self):
        """Release the HTTP session."""
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

# ═══════════════════════════════════════════════════════════════════════
# Convenience: build a policy from a config dict
# ═══════════════════════════════════════════════════════════════════════

def build_policy(config: dict) -> Callable[[str],
                                              Awaitable[
                                                  tuple[str, list[int], list[float]]]]:
    """Construct a policy from a flat config dict.

    Used by ``scripts/rl_demo.py`` and elsewhere to make policy choice
    declarative. Keys::

        kind: "mock" | "openai" | "callable"
        # mock:
        tactics: [...]    seed: 0    shuffle: false
        # openai:
        base_url: "http://localhost:8001/v1"
        model: "deepseek-ai/DeepSeek-Prover-
        api_key: "..."
        # callable:
        fn: callable
    """
    kind = config.get("kind", "mock").lower()
    if kind == "mock":
        return MockPolicy(
            tactics=config.get("tactics"),
            seed=config.get("seed", 0),
            shuffle=config.get("shuffle", False),
            token_ids_for=config.get("token_ids_for"),
        )
    if kind == "openai":
        return OpenAIPolicy(
            base_url=config.get("base_url"),
            model=config.get("model", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            tokenizer=config.get("tokenizer"),
            temperature=config.get("temperature", 0.9),
            max_tokens=config.get("max_tokens", 256),
            timeout_s=config.get("timeout_s", 60.0),
            logprobs_for_unsourced=config.get(
                "logprobs_for_unsourced", False),
        )
    if kind == "callable":
        fn = config.get("fn")
        if not callable(fn):
            raise ValueError("policy.kind=callable requires a `fn` callable")
        return CallablePolicy(fn)
    raise ValueError(f"Unknown policy kind: {kind!r}")

"""agent/brain/claude_provider.py — Claude 专用 Provider (支持 Tool Use + Extended Thinking)"""
from __future__ import annotations
import os, time, logging, random, threading
from agent.brain.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = "",
                 extended_thinking: bool = False):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._extended_thinking = extended_thinking
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        """Thread-safe lazy singleton client — reuses connection pool."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:  # double-checked locking
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str = "", user: str = "", temperature: float = 0.7,
                 tools: list = None, max_tokens: int = 4096) -> LLMResponse:
        return self.chat(system, [{"role": "user", "content": user}],
                         temperature, tools, max_tokens)

    def chat(self, system: str = "", messages: list[dict] = None,
             temperature: float = 0.7, tools: list = None,
             max_tokens: int = 4096) -> LLMResponse:
        """Multi-turn chat supporting tool_use and extended thinking."""
        client = self._get_client()
        kwargs = dict(model=self._model, max_tokens=max_tokens,
                      system=system, messages=messages or [])

        # Extended thinking requires temperature=1 and uses budget_tokens
        # instead of max_tokens for the thinking portion.
        if self._extended_thinking:
            kwargs["temperature"] = 1
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(max_tokens, 10000),
            }
        else:
            kwargs["temperature"] = temperature

        if tools:
            kwargs["tools"] = tools

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                start = time.time()
                response = client.messages.create(**kwargs)
                latency = int((time.time() - start) * 1000)

                content = ""
                tool_calls = []
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "thinking":
                        pass  # thinking blocks are internal, not exposed
                    elif block.type == "tool_use":
                        tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
                return LLMResponse(
                    content=content, model=self._model,
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    latency_ms=latency, tool_calls=tool_calls,
                    stop_reason=response.stop_reason or "end_turn")
            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    backoff = min(_INITIAL_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                    jitter = random.uniform(0, backoff * 0.3)
                    wait = backoff + jitter
                    logger.warning(
                        f"Claude API call failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}): "
                        f"{e}. Retrying in {wait:.1f}s...")
                    # Use time.sleep here (sync provider). The async provider
                    # (AsyncLLMProvider) uses asyncio.sleep instead.
                    time.sleep(wait)
                else:
                    logger.error(f"Claude API call failed after {_MAX_RETRIES + 1} attempts: {e}")

        raise last_error


class MockProvider(LLMProvider):
    @property
    def model_name(self) -> str: return "mock"
    def generate(self, system="", user="", temperature=0.7, tools=None, max_tokens=4096):
        return self.chat(system, [{"role": "user", "content": user}],
                         temperature, tools, max_tokens)
    def chat(self, system="", messages=None, temperature=0.7, tools=None, max_tokens=4096):
        return LLMResponse(content="```lean\n:= by\n  sorry\n```",
                           model="mock", tokens_in=100, tokens_out=20,
                           latency_ms=50, stop_reason="end_turn")

def create_provider(config: dict) -> LLMProvider:
    p = config.get("provider", "anthropic")
    if p == "anthropic":
        return ClaudeProvider(model=config.get("model", "claude-sonnet-4-20250514"),
                              api_key=config.get("api_key", ""),
                              extended_thinking=config.get("extended_thinking", False))
    elif p == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown provider: {p}")

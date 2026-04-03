"""agent/brain/claude_provider.py — Claude 专用 Provider (支持 Tool Use + Extended Thinking)"""
from __future__ import annotations
import os, time, logging
from agent.brain.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = "",
                 extended_thinking: bool = False):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._extended_thinking = extended_thinking

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float = 0.7,
                 tools: list = None, max_tokens: int = 4096) -> LLMResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        start = time.time()
        kwargs = dict(model=self._model, max_tokens=max_tokens, temperature=temperature,
                      system=system, messages=[{"role": "user", "content": user}])
        if tools:
            kwargs["tools"] = tools
        response = client.messages.create(**kwargs)
        latency = int((time.time() - start) * 1000)
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "input": block.input, "id": block.id})
        return LLMResponse(content=content, model=self._model,
                           tokens_in=response.usage.input_tokens,
                           tokens_out=response.usage.output_tokens,
                           latency_ms=latency, tool_calls=tool_calls)

class MockProvider(LLMProvider):
    @property
    def model_name(self) -> str: return "mock"
    def generate(self, system="", user="", temperature=0.7, tools=None, max_tokens=4096):
        return LLMResponse(content="```lean\n:= by\n  sorry\n```",
                           model="mock", tokens_in=100, tokens_out=20, latency_ms=50)

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

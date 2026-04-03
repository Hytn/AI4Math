"""agent/brain/llm_provider.py — LLM Provider 抽象基类"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    tool_calls: list = None
    raw_response: Optional[dict] = None

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.7,
                 tools: list = None, max_tokens: int = 4096) -> LLMResponse: ...
    @property
    @abstractmethod
    def model_name(self) -> str: ...

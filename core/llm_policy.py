"""
core/llm_policy.py — LLM 策略调用层

职责：把"生成 Lean proof"这件事抽象为统一接口，底层可替换不同 LLM。
支持：OpenAI (GPT-4o)、Anthropic (Claude)、本地模型 (OpenAI-compatible API)。

设计要点：
  - 策略角色 (role) 分离：proof_generator / proof_patcher / lemma_proposer
  - prompt 模板化，上层只传入结构化信息
  - token 用量和延迟追踪
"""

from __future__ import annotations

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 调用结果"""
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw_response: Optional[dict] = None


# ── Prompt 模板 ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Lean 4 theorem prover. Your task is to generate correct, \
compilable Lean 4 proofs that pass the Lean kernel.

Key rules:
1. Output ONLY the proof body (starting with `:= by` or `:= ...`). Do not repeat the theorem statement.
2. Use tactics from Mathlib when appropriate (simp, ring, linarith, omega, norm_num, exact?, apply?, etc.).
3. Prefer simple, direct proofs over complex ones.
4. When in doubt, break the proof into smaller `have` steps.
5. Do NOT use `sorry` — every goal must be closed.
6. Output the proof inside a ```lean code block.
"""

FIRST_ATTEMPT_TEMPLATE = """\
Prove the following Lean 4 theorem. The theorem uses Mathlib.

## Theorem
```lean
{theorem_statement}
```
{premises_section}
Generate a complete proof. Output only the proof body inside ```lean blocks.
"""

RETRY_TEMPLATE = """\
Prove the following Lean 4 theorem. Your previous attempt(s) had errors.

## Theorem
```lean
{theorem_statement}
```
{premises_section}
{error_analysis}
{error_history}
Generate a corrected proof. Output only the proof body inside ```lean blocks.
"""


def _format_premises_section(premises: list[str]) -> str:
    if not premises:
        return ""
    premise_list = "\n".join(f"- `{p}`" for p in premises[:20])
    return f"\n## Potentially useful Mathlib lemmas\n{premise_list}\n"


def build_prompt(
    theorem_statement: str,
    error_analysis: str = "",
    error_history: str = "",
    premises: list[str] | None = None,
) -> str:
    """构建 LLM prompt"""
    premises_section = _format_premises_section(premises or [])

    if error_analysis:
        return RETRY_TEMPLATE.format(
            theorem_statement=theorem_statement,
            premises_section=premises_section,
            error_analysis=error_analysis,
            error_history=error_history,
        )
    else:
        return FIRST_ATTEMPT_TEMPLATE.format(
            theorem_statement=theorem_statement,
            premises_section=premises_section,
        )


def extract_lean_code(response: str) -> str:
    """从 LLM 输出中提取 Lean 代码块"""
    # 尝试提取 ```lean ... ``` 块
    import re
    pattern = r"```lean\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # 返回最后一个代码块（通常是最终版本）
        return matches[-1].strip()

    # 尝试提取 ``` ... ``` 块
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # 兜底：直接返回原文（去掉明显的非代码部分）
    lines = response.strip().split("\n")
    code_lines = [l for l in lines if not l.startswith("**") and not l.startswith("##")]
    return "\n".join(code_lines).strip()


# ── LLM Provider 抽象 ──────────────────────────────────────────

class LLMProvider(ABC):
    """LLM 提供者的抽象基类"""

    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.7) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API"""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = ""):
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set")

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float = 0.7) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        start = time.time()

        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        latency = int((time.time() - start) * 1000)
        content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            model=self._model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            latency_ms=latency,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI API (也兼容 vLLM/ollama 等 OpenAI-compatible 接口)"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = "",
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        if not self._api_key:
            logger.warning("OPENAI_API_KEY not set")

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, system: str, user: str, temperature: float = 0.7) -> LLMResponse:
        import openai

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url

        client = openai.OpenAI(**kwargs)
        start = time.time()

        response = client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        latency = int((time.time() - start) * 1000)
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=self._model,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )


class MockProvider(LLMProvider):
    """模拟 LLM，用于不消耗 API 的本地测试"""

    def __init__(self):
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "mock"

    def generate(self, system: str, user: str, temperature: float = 0.7) -> LLMResponse:
        self._call_count += 1
        # 返回一个简单但格式正确的 proof 尝试
        mock_proof = """\
```lean
:= by
  sorry
```
"""
        return LLMResponse(
            content=mock_proof,
            model="mock",
            tokens_in=len(system + user) // 4,
            tokens_out=20,
            latency_ms=50,
        )


# ── 工厂函数 ───────────────────────────────────────────────────

def create_provider(config: dict) -> LLMProvider:
    """根据配置创建 LLM provider"""
    provider_type = config.get("provider", "anthropic")

    if provider_type == "anthropic":
        return AnthropicProvider(
            model=config.get("model", "claude-sonnet-4-20250514"),
            api_key=config.get("api_key", ""),
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            model=config.get("model", "gpt-4o"),
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url", ""),
        )
    elif provider_type == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}")

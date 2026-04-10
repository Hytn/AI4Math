"""agent/runtime/agent_loop.py — Multi-turn agent loop with tool use

Inspired by Claude Code's QueryEngine: the agent enters a loop where
the LLM can call tools, receive results, and reason further until it
produces a final answer or exhausts its budget.

This is the core upgrade that transforms SubAgent from a "single-shot
generator" into an "autonomous reasoning agent".

Flow::

    System prompt + context
              │
              ▼
    ┌──── LLM call ◄───────────────────────────────┐
    │         │                                      │
    │    ┌────┴────┐                                 │
    │    │ Response │                                 │
    │    └────┬────┘                                 │
    │    text_only?──yes──► return final result       │
    │         │no                                    │
    │    tool_use blocks                             │
    │         │                                      │
    │    ┌────┴────┐                                 │
    │    │ Execute  │  (parallel if multiple)         │
    │    │  tools   │                                 │
    │    └────┬────┘                                 │
    │         │                                      │
    │    tool results → append to messages ───────────┘
    │
    └── max_turns reached → return best so far

Usage::

    loop = AgentLoop(llm=provider, tools=registry)
    result = await loop.run(
        system_prompt="You are a Lean4 prover...",
        initial_message="Prove: theorem foo ...",
        max_turns=10,
    )
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from agent.brain.async_llm_provider import AsyncLLMProvider
from agent.brain.llm_provider import LLMResponse
from agent.tools.base import ToolContext, ToolResult
from agent.tools.registry import ToolRegistry
from common.response_parser import extract_lean_code
from prover.verifier.sorry_detector import detect_sorry

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the agent loop."""
    max_turns: int = 10                # Max LLM calls in one loop
    max_tokens_per_turn: int = 4096
    temperature: float = 0.7
    timeout_seconds: float = 120.0     # Total loop timeout
    tool_timeout_seconds: float = 30.0
    # Stop conditions
    stop_on_proof: bool = True         # Stop when Lean code w/o sorry found
    stop_on_text_only: bool = True     # Stop when LLM returns no tool calls
    # Budget
    max_total_tokens: int = 200_000


@dataclass
class LoopMessage:
    """A message in the agent conversation."""
    role: str           # "user", "assistant", "tool_result"
    content: str = ""
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    tokens: int = 0


@dataclass
class LoopResult:
    """Result of the agent loop."""
    content: str                        # Final text from LLM
    proof_code: str = ""               # Extracted Lean code
    messages: list[LoopMessage] = field(default_factory=list)
    turns_used: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    tools_called: list[str] = field(default_factory=list)
    stopped_reason: str = ""           # "proof_found", "text_only", "max_turns", "timeout", "error"

    @property
    def has_proof(self) -> bool:
        return bool(self.proof_code.strip())


class AgentLoop:
    """Multi-turn agent loop with tool use.

    The LLM and tools form a feedback cycle: the LLM reasons about the
    theorem, calls tools to search premises / inspect goals / verify
    partial proofs, and uses the results to refine its approach.
    """

    def __init__(
        self,
        llm: AsyncLLMProvider,
        tools: ToolRegistry,
        config: LoopConfig = None,
        on_turn: Optional[Callable] = None,
    ):
        self.llm = llm
        self.tools = tools
        self.config = config or LoopConfig()
        self.on_turn = on_turn  # callback(turn_number, message)

    async def run(
        self,
        system_prompt: str,
        initial_message: str,
        injected_context: str = "",
        tool_ctx: ToolContext = None,
    ) -> LoopResult:
        """Run the agent loop.

        Args:
            system_prompt: System prompt for the LLM
            initial_message: First user message (task description)
            injected_context: Additional context prepended to first message
            tool_ctx: Shared context for tool execution

        Returns:
            LoopResult with final content, proof code, and conversation history
        """
        config = self.config
        tool_ctx = tool_ctx or ToolContext()
        start_time = time.time()

        # Build initial message
        full_initial = initial_message
        if injected_context:
            full_initial = f"{injected_context}\n\n{initial_message}"

        # Conversation state — Claude API message format
        messages = [{"role": "user", "content": full_initial}]
        history: list[LoopMessage] = [
            LoopMessage(role="user", content=full_initial)
        ]

        # Tool schemas
        tool_schemas = self.tools.to_claude_tools_schema(
            permission_filter=tool_ctx.allowed_permissions)

        total_tokens = 0
        tools_called = []
        last_content = ""
        last_proof = ""

        for turn in range(config.max_turns):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > config.timeout_seconds:
                return self._make_result(
                    last_content, last_proof, history, turn,
                    total_tokens, start_time, tools_called, "timeout")

            # Check token budget
            if total_tokens >= config.max_total_tokens:
                return self._make_result(
                    last_content, last_proof, history, turn,
                    total_tokens, start_time, tools_called, "token_budget")

            # ── LLM call ──
            try:
                resp = await self._call_llm_with_messages(
                    system_prompt, messages, tool_schemas, config)

            except Exception as e:
                logger.error(f"LLM call failed on turn {turn}: {e}")
                return self._make_result(
                    last_content, last_proof, history, turn,
                    total_tokens, start_time, tools_called,
                    f"error: {e}")

            total_tokens += resp.tokens_in + resp.tokens_out

            # ── Parse response ──
            content = resp.content
            tool_calls = resp.tool_calls or []
            last_content = content

            # Extract proof if present
            proof = extract_lean_code(content)
            if proof:
                last_proof = proof

            # Record in history
            history.append(LoopMessage(
                role="assistant", content=content,
                tool_calls=tool_calls,
                tokens=resp.tokens_in + resp.tokens_out))

            # Callback
            if self.on_turn:
                try:
                    self.on_turn(turn, history[-1])
                except Exception as e:
                    logger.debug(f"on_turn callback failed: {e}")

            # ── Check stop conditions ──

            # Stop if proof found (no sorry)
            if (config.stop_on_proof and proof
                    and detect_sorry(proof).is_clean):
                return self._make_result(
                    content, proof, history, turn + 1,
                    total_tokens, start_time, tools_called, "proof_found")

            # Stop if no tool calls (LLM gave final answer)
            if not tool_calls and config.stop_on_text_only:
                return self._make_result(
                    content, last_proof, history, turn + 1,
                    total_tokens, start_time, tools_called, "text_only")

            if not tool_calls:
                # No tools and not stopping → just continue
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Continue. If you have a proof, output it in a ```lean block.",
                })
                continue

            # ── Execute tool calls ──
            messages.append({"role": "assistant", "content": self._build_assistant_content(resp)})

            tool_results = await self._execute_tools(tool_calls, tool_ctx)
            tools_called.extend(tc["name"] for tc in tool_calls)

            # Append tool results using proper Claude API format
            tool_result_blocks = []
            for tc, tr in zip(tool_calls, tool_results):
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc.get("id", f"tool_{tc['name']}"),
                    "content": tr["content"],
                    "is_error": tr.get("is_error", False),
                })
            messages.append({
                "role": "user",
                "content": tool_result_blocks,
            })

            history.append(LoopMessage(
                role="tool_result",
                tool_results=[tr["content"] for tr in tool_results]))

        # Max turns reached
        return self._make_result(
            last_content, last_proof, history, config.max_turns,
            total_tokens, start_time, tools_called, "max_turns")

    async def _call_llm_with_messages(
        self,
        system: str,
        messages: list[dict],
        tool_schemas: list[dict],
        config: LoopConfig,
    ) -> LLMResponse:
        """Call LLM with full message history using chat() API.

        Uses the provider's chat() method which supports proper multi-turn
        conversations with the Claude messages API.
        """
        # Use native chat() if available (preferred path)
        if hasattr(self.llm, 'chat'):
            if asyncio.iscoroutinefunction(self.llm.chat):
                return await self.llm.chat(
                    system=system,
                    messages=messages,
                    temperature=config.temperature,
                    tools=tool_schemas if tool_schemas else None,
                    max_tokens=config.max_tokens_per_turn,
                )
            else:
                # Sync provider — run in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self.llm.chat(
                        system=system,
                        messages=messages,
                        temperature=config.temperature,
                        tools=tool_schemas if tool_schemas else None,
                        max_tokens=config.max_tokens_per_turn,
                    ))

        # Fallback: concatenate messages for providers without chat()
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"[{role}]\n{content}")
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            text_parts.append(
                                f"[calling tool: {block.get('name')}]")
                        elif block.get("type") == "tool_result":
                            text_parts.append(block.get("content", ""))
                    else:
                        text_parts.append(str(block))
                parts.append(f"[{role}]\n" + "\n".join(text_parts))

        combined = "\n\n".join(parts)
        return await self.llm.generate(
            system=system,
            user=combined,
            temperature=config.temperature,
            tools=tool_schemas if tool_schemas else None,
            max_tokens=config.max_tokens_per_turn,
        )

    def _build_assistant_content(self, resp: LLMResponse) -> str | list:
        """Build assistant message content including tool use blocks.

        Uses proper Claude API tool_use format when tool calls are present.
        Falls back to text-only string when no tool calls exist.
        """
        if not resp.tool_calls:
            return resp.content or ""

        blocks = []
        if resp.content:
            blocks.append({"type": "text", "text": resp.content})
        for tc in resp.tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"tool_{tc['name']}"),
                "name": tc["name"],
                "input": tc.get("input", {}),
            })
        return blocks

    async def _execute_tools(
        self,
        tool_calls: list[dict],
        ctx: ToolContext,
    ) -> list[dict]:
        """Execute tool calls (parallel when possible)."""
        tasks = []
        for tc in tool_calls:
            name = tc.get("name", "")
            input_data = tc.get("input", {})
            tasks.append(self.tools.execute(name, input_data, ctx))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        formatted = []
        for tc, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                formatted.append({
                    "tool_use_id": tc.get("id", ""),
                    "content": f"Error: {result}",
                    "is_error": True,
                })
            else:
                formatted.append({
                    "tool_use_id": tc.get("id", ""),
                    "content": result.content,
                    "is_error": result.is_error,
                })
        return formatted

    def _make_result(
        self,
        content: str,
        proof: str,
        history: list[LoopMessage],
        turns: int,
        tokens: int,
        start_time: float,
        tools_called: list[str],
        reason: str,
    ) -> LoopResult:
        return LoopResult(
            content=content,
            proof_code=proof,
            messages=history,
            turns_used=turns,
            total_tokens=tokens,
            total_latency_ms=int((time.time() - start_time) * 1000),
            tools_called=tools_called,
            stopped_reason=reason,
        )

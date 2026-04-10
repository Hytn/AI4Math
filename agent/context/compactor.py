"""agent/context/compactor.py — LLM-based context compaction

Inspired by Claude Code's compact system (services/compact/compact.ts):
  - When context window fills up, use an LLM to summarize older messages
  - Preserves key information (theorem, discoveries, errors)
  - Multiple strategies: LLM summary, micro-compact, selective drop

This goes beyond the simple truncation in ContextWindow._compress() by
using the LLM to produce intelligent summaries that retain proof-relevant
information while dramatically reducing token count.

Usage::

    compactor = ContextCompactor(llm=provider)

    # Compact a message history
    result = await compactor.compact(
        messages=conversation_history,
        keep_recent=3,         # Keep last 3 messages verbatim
        target_tokens=5000,    # Compress to ~5000 tokens
    )

    # Use result.compacted_messages as new conversation
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.context.context_window import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class CompactionResult:
    """Result of context compaction."""
    compacted_messages: list[dict]  # New message list (summary + recent)
    summary_text: str               # The generated summary
    original_count: int = 0         # Messages before compaction
    compacted_count: int = 0        # Messages after compaction
    tokens_before: int = 0
    tokens_after: int = 0
    latency_ms: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0
        return 1.0 - (self.tokens_after / self.tokens_before)


COMPACTION_SYSTEM_PROMPT = """\
You are a proof assistant context compressor. Your job is to summarize \
a conversation history between a mathematician and a theorem prover.

RULES:
1. Preserve ALL theorem statements, goal states, and proven lemmas exactly.
2. Preserve key error messages and their fixes.
3. Preserve any proof code that partially worked.
4. Drop redundant attempts that were superseded by later ones.
5. Drop verbose tool outputs, keep only conclusions.
6. Keep the summary under {target_tokens} tokens.
7. Use structured format with clear sections.

Output format:
## Theorem
[exact theorem statement]

## Key Findings
- [finding 1]
- [finding 2]

## Proven Lemmas
[any intermediate lemmas that were verified]

## Failed Approaches
- [approach]: [why it failed]

## Current State
[what has been achieved, what remains]
"""


class ContextCompactor:
    """LLM-based context compaction.

    Three compaction strategies (applied in escalation order):

    1. **Micro-compact**: Drop tool results and keep only summaries.
       Cheapest, preserves conversation structure.

    2. **LLM Summary**: Use the LLM to generate an intelligent summary
       of older messages. Best quality, costs one LLM call.

    3. **Aggressive Drop**: Remove all messages except the most recent N.
       Last resort, used when budget is critically low.
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    async def compact(
        self,
        messages: list[dict],
        keep_recent: int = 3,
        target_tokens: int = 5000,
        strategy: str = "auto",
    ) -> CompactionResult:
        """Compact conversation history.

        Args:
            messages: Full conversation history
            keep_recent: Number of recent messages to keep verbatim
            target_tokens: Target token count for compacted history
            strategy: "auto", "micro", "llm_summary", "aggressive_drop"

        Returns:
            CompactionResult with compacted messages
        """
        start = time.time()
        tokens_before = sum(
            estimate_tokens(self._msg_to_text(m)) for m in messages)

        if len(messages) <= keep_recent + 1:
            # Not enough to compact
            return CompactionResult(
                compacted_messages=messages,
                summary_text="",
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # Split into old (to compact) and recent (to keep)
        old_messages = messages[:-keep_recent] if keep_recent > 0 else messages
        recent_messages = messages[-keep_recent:] if keep_recent > 0 else []

        # Choose strategy
        if strategy == "auto":
            old_tokens = sum(
                estimate_tokens(self._msg_to_text(m)) for m in old_messages)
            if old_tokens < target_tokens:
                # Already small enough, just micro-compact
                strategy = "micro"
            elif self._llm is not None:
                strategy = "llm_summary"
            else:
                strategy = "micro"

        # Apply strategy
        if strategy == "llm_summary":
            summary = await self._llm_summarize(old_messages, target_tokens)
        elif strategy == "micro":
            summary = self._micro_compact(old_messages, target_tokens)
        elif strategy == "aggressive_drop":
            summary = self._aggressive_summary(old_messages)
        else:
            summary = self._micro_compact(old_messages, target_tokens)

        # Build compacted messages
        compacted = [
            {"role": "user", "content": f"[Context Summary]\n{summary}"}
        ] + recent_messages

        tokens_after = sum(
            estimate_tokens(self._msg_to_text(m)) for m in compacted)

        return CompactionResult(
            compacted_messages=compacted,
            summary_text=summary,
            original_count=len(messages),
            compacted_count=len(compacted),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            latency_ms=int((time.time() - start) * 1000),
        )

    async def _llm_summarize(
        self,
        messages: list[dict],
        target_tokens: int,
    ) -> str:
        """Use the LLM to generate an intelligent summary."""
        # Build conversation text for summarization
        conv_text = self._format_for_summary(messages)

        system = COMPACTION_SYSTEM_PROMPT.format(target_tokens=target_tokens)
        user = f"Summarize this proof conversation:\n\n{conv_text}"

        try:
            if asyncio.iscoroutinefunction(getattr(self._llm, 'generate', None)):
                resp = await self._llm.generate(
                    system=system, user=user,
                    temperature=0.0, max_tokens=target_tokens * 2)
            elif hasattr(self._llm, 'generate'):
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None, lambda: self._llm.generate(
                        system=system, user=user,
                        temperature=0.0, max_tokens=target_tokens * 2))
            else:
                return self._micro_compact(messages, target_tokens)

            return resp.content

        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}, falling back to micro-compact")
            return self._micro_compact(messages, target_tokens)

    def _micro_compact(
        self,
        messages: list[dict],
        target_tokens: int,
    ) -> str:
        """Cheap compaction: extract key info without LLM call."""
        sections = {
            "theorem": "",
            "findings": [],
            "errors": [],
            "proofs": [],
        }

        for msg in messages:
            text = self._msg_to_text(msg)
            role = msg.get("role", "")

            # Extract theorem statement
            if "theorem" in text.lower() and not sections["theorem"]:
                for line in text.split("\n"):
                    if line.strip().startswith("theorem ") or line.strip().startswith("lemma "):
                        sections["theorem"] = line.strip()
                        break

            # Extract proof code
            import re
            lean_blocks = re.findall(r'```lean\s*\n(.*?)```', text, re.DOTALL)
            for block in lean_blocks:
                block = block.strip()
                if block and "sorry" not in block and len(block) > 10:
                    sections["proofs"].append(block[:300])

            # Extract errors (keep short)
            if "error" in text.lower() or "failed" in text.lower():
                for line in text.split("\n"):
                    if ("error" in line.lower() or "failed" in line.lower()):
                        err = line.strip()[:150]
                        if err and err not in sections["errors"]:
                            sections["errors"].append(err)
                            if len(sections["errors"]) > 5:
                                break

            # Extract findings from tool results
            if role == "user" and "[Tool result" in text:
                # Keep just the first line of tool results
                for line in text.split("\n")[1:3]:
                    if line.strip():
                        sections["findings"].append(line.strip()[:150])

        # Build summary
        parts = []
        if sections["theorem"]:
            parts.append(f"Theorem: {sections['theorem']}")
        if sections["proofs"]:
            parts.append("Best proof attempts:")
            for p in sections["proofs"][-2:]:  # Keep last 2
                parts.append(f"```lean\n{p}\n```")
        if sections["errors"]:
            parts.append("Key errors encountered:")
            for e in sections["errors"][-3:]:
                parts.append(f"- {e}")
        if sections["findings"]:
            parts.append("Findings:")
            for f in sections["findings"][-5:]:
                parts.append(f"- {f}")

        summary = "\n".join(parts)

        # Truncate if still too long
        max_chars = target_tokens * 4
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n... (truncated)"

        return summary

    def _aggressive_summary(self, messages: list[dict]) -> str:
        """Last-resort: just keep theorem and latest proof attempt."""
        theorem = ""
        last_proof = ""
        for msg in messages:
            text = self._msg_to_text(msg)
            for line in text.split("\n"):
                if line.strip().startswith("theorem ") or line.strip().startswith("lemma "):
                    theorem = line.strip()
            import re
            blocks = re.findall(r'```lean\s*\n(.*?)```', text, re.DOTALL)
            if blocks:
                last_proof = blocks[-1].strip()

        return f"Theorem: {theorem}\nLast attempt:\n```lean\n{last_proof}\n```"

    def _format_for_summary(self, messages: list[dict]) -> str:
        """Format messages for LLM summarization input."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            text = self._msg_to_text(msg)
            # Truncate very long messages
            if len(text) > 2000:
                text = text[:1000] + "\n...\n" + text[-500:]
            parts.append(f"[{role}]: {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _msg_to_text(msg: dict) -> str:
        """Extract text content from a message."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(block.get("content", ""))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)

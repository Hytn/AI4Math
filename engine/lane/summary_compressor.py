"""engine/lane/summary_compressor.py — Lean error & context compression

Inspired by claw-code's SummaryCompressionBudget / compress_summary_text().

Problem: Raw Lean 4 errors are verbose and repetitive. Injecting them uncompressed
into repair prompts wastes tokens and introduces noise. Broadcast messages from
multiple directions similarly bloat the context window.

This module provides three compressors:

1. compress_lean_errors   — Deduplicate, categorize, and truncate Lean errors
2. compress_feedback      — Compress AgentFeedback.to_prompt() output
3. compress_broadcast     — Compress cross-direction broadcast messages

Design constraints:
  - Output must remain LLM-readable (not machine-only)
  - Preserve the most actionable information
  - Hard budget on output chars (default 1200, matching claw-code)
  - Dedup identical/near-identical error lines
  - Preserve error category distribution for diagnosis

Usage::

    from engine.lane.summary_compressor import compress_lean_errors, compress_feedback

    # Before injecting into repair prompt:
    compressed = compress_lean_errors(raw_errors, budget=1200)

    # Before injecting AgentFeedback:
    compressed = compress_feedback(feedback_text, budget=800)
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompressionBudget:
    """Budget for summary compression — mirrors claw-code's SummaryCompressionBudget."""
    max_chars: int = 1200
    max_lines: int = 24
    max_line_chars: int = 160


@dataclass
class CompressionResult:
    """Result of compression with metrics — mirrors claw-code's SummaryCompressionResult."""
    summary: str
    original_chars: int
    compressed_chars: int
    original_lines: int
    compressed_lines: int
    removed_duplicate_lines: int
    truncated: bool

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 1.0
        return self.compressed_chars / self.original_chars


# ═══════════════════════════════════════════════════════════════════════════
# 1. Lean Error Compression
# ═══════════════════════════════════════════════════════════════════════════

def compress_lean_errors(
    errors: list[str] | str,
    budget: int = 1200,
    max_lines: int = 24,
) -> str:
    """Compress Lean 4 compiler errors for LLM consumption.

    Strategy:
      1. Normalize whitespace and split into individual error blocks
      2. Deduplicate identical and near-identical errors
      3. Categorize errors and show distribution summary
      4. Keep the most diverse set of errors within budget
      5. Truncate individual long errors

    Args:
        errors: Raw error text (str) or list of error strings.
        budget: Maximum output characters.
        max_lines: Maximum output lines.

    Returns:
        Compressed error summary, LLM-readable.
    """
    if isinstance(errors, str):
        error_list = _split_lean_errors(errors)
    else:
        error_list = list(errors)

    if not error_list:
        return ""

    # Normalize
    normalized = [_normalize_error(e) for e in error_list if e.strip()]
    if not normalized:
        return ""

    # Deduplicate (preserve order, count occurrences)
    seen: OrderedDict[str, int] = OrderedDict()
    dedup_key_map: dict[str, str] = {}  # dedup_key → first full error
    for err in normalized:
        key = _dedup_key(err)
        if key in seen:
            seen[key] += 1
        else:
            seen[key] = 1
            dedup_key_map[key] = err

    unique_errors = list(dedup_key_map.values())
    total_count = len(normalized)
    unique_count = len(unique_errors)
    dup_count = total_count - unique_count

    # Categorize
    categories = Counter()
    for err in unique_errors:
        cat = _categorize_error(err)
        categories[cat] += 1

    # Build output
    parts = []

    # Header: distribution summary
    if total_count > 1:
        cat_str = ", ".join(f"{cat}: {cnt}" for cat, cnt in categories.most_common(5))
        header = f"[{total_count} errors, {unique_count} unique] Categories: {cat_str}"
        if dup_count > 0:
            header += f" ({dup_count} duplicates removed)"
        parts.append(header)
        parts.append("")

    # Select diverse errors within budget
    remaining_budget = budget - sum(len(p) + 1 for p in parts)
    selected = _select_diverse_errors(unique_errors, seen, remaining_budget, max_lines - 3)

    for err, count in selected:
        line = _truncate_line(err, 160)
        if count > 1:
            line = f"{line} (×{count})"
        parts.append(line)

    result = "\n".join(parts)

    # Final truncation
    if len(result) > budget:
        result = result[:budget - 20] + "\n... (truncated)"

    return result


def _split_lean_errors(text: str) -> list[str]:
    """Split a raw Lean error output into individual error blocks."""
    # Lean errors typically separated by blank lines or file:line:col patterns
    blocks = re.split(r'\n(?=\S+:\d+:\d+:)', text)
    if len(blocks) <= 1:
        # Try splitting by blank lines
        blocks = re.split(r'\n\s*\n', text)
    return [b.strip() for b in blocks if b.strip()]


def _normalize_error(error: str) -> str:
    """Normalize whitespace and remove noisy prefixes."""
    # Remove file paths (keep just the error content)
    error = re.sub(r'^[^\s]+\.lean:\d+:\d+:\s*', '', error, count=1)
    # Collapse multiple spaces/tabs
    error = re.sub(r'[ \t]+', ' ', error)
    # Collapse multiple newlines
    error = re.sub(r'\n{3,}', '\n\n', error)
    return error.strip()


def _dedup_key(error: str) -> str:
    """Generate a deduplication key for an error.

    Ignores line numbers, specific variable names in common positions,
    and whitespace differences.
    """
    key = error.lower()
    # Remove line/column numbers
    key = re.sub(r'\d+:\d+', 'N:N', key)
    # Normalize specific identifiers in common positions
    key = re.sub(r"'[a-z_][a-z0-9_]*'", "'ID'", key)
    # Remove excess whitespace
    key = re.sub(r'\s+', ' ', key).strip()
    # Only use first 200 chars for comparison
    return key[:200]


def _categorize_error(error: str) -> str:
    """Categorize a Lean error into a human-readable category."""
    lower = error.lower()
    if 'type mismatch' in lower:
        return 'type_mismatch'
    if 'unknown identifier' in lower or 'unknown constant' in lower:
        return 'unknown_id'
    if "tactic" in lower and "failed" in lower:
        return 'tactic_failed'
    if 'unsolved goals' in lower:
        return 'unsolved_goals'
    if 'expected' in lower and 'token' in lower:
        return 'syntax'
    if 'sorry' in lower:
        return 'sorry'
    if 'timeout' in lower or 'heartbeat' in lower:
        return 'timeout'
    if 'import' in lower and ('not found' in lower or 'unknown' in lower):
        return 'import'
    if 'failed to synthesize' in lower:
        return 'instance'
    return 'other'


def _select_diverse_errors(
    unique_errors: list[str],
    count_map: OrderedDict[str, int],
    budget: int,
    max_count: int,
) -> list[tuple[str, int]]:
    """Select a diverse set of errors within budget.

    Prioritizes:
    1. Errors from different categories (diversity)
    2. Errors with higher occurrence count (frequency)
    3. Shorter errors (information density)
    """
    if not unique_errors:
        return []

    # Score each error
    scored = []
    seen_cats = set()
    for err in unique_errors:
        cat = _categorize_error(err)
        key = _dedup_key(err)
        count = count_map.get(key, 1)
        diversity_bonus = 10 if cat not in seen_cats else 0
        seen_cats.add(cat)
        score = diversity_bonus + count * 2 + (1.0 / max(1, len(err)))
        scored.append((score, err, count))

    scored.sort(key=lambda x: -x[0])

    # Select within budget
    selected = []
    used_chars = 0
    for _, err, count in scored:
        line_len = min(len(err), 160) + (len(f" (×{count})") if count > 1 else 0) + 1
        if used_chars + line_len > budget:
            continue
        if len(selected) >= max_count:
            break
        selected.append((err, count))
        used_chars += line_len

    return selected


def _truncate_line(text: str, max_chars: int) -> str:
    """Truncate a single line, keeping the most informative part."""
    if len(text) <= max_chars:
        return text
    # For multi-line errors, keep first and last line
    lines = text.split('\n')
    if len(lines) >= 3:
        first = lines[0][:max_chars // 2]
        last = lines[-1][:max_chars // 2]
        return f"{first}\n  ... ({len(lines) - 2} lines omitted)\n  {last}"
    return text[:max_chars - 15] + " ... (truncated)"


# ═══════════════════════════════════════════════════════════════════════════
# 2. AgentFeedback Compression
# ═══════════════════════════════════════════════════════════════════════════

def compress_feedback(
    feedback_text: str,
    budget: int = 800,
) -> str:
    """Compress AgentFeedback.to_prompt() output.

    Removes redundant sections, deduplicates goal descriptions,
    and truncates verbose type signatures.
    """
    if not feedback_text or len(feedback_text) <= budget:
        return feedback_text

    lines = feedback_text.split('\n')

    # Phase 1: Remove blank lines and normalize
    lines = [l for l in lines if l.strip()]

    # Phase 2: Truncate long type signatures
    compressed = []
    for line in lines:
        # Lean type sigs can be very long
        if ('Expected:' in line or 'Actual:' in line or '⊢' in line):
            line = _truncate_line(line, 120)
        compressed.append(line)

    # Phase 3: Deduplicate goals
    seen_goals = set()
    final = []
    for line in compressed:
        # Detect goal lines (typically start with ⊢ or contain "goal")
        if '⊢' in line:
            key = re.sub(r'\s+', ' ', line.strip())
            if key in seen_goals:
                continue
            seen_goals.add(key)
        final.append(line)

    result = '\n'.join(final)

    # Final truncation
    if len(result) > budget:
        result = result[:budget - 20] + "\n... (truncated)"

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. Broadcast Message Compression
# ═══════════════════════════════════════════════════════════════════════════

def compress_broadcast(
    messages: list[dict] | str,
    budget: int = 1500,
) -> str:
    """Compress cross-direction broadcast messages.

    Strategy:
      - Deduplicate similar discoveries
      - Prioritize positive discoveries over negative knowledge
      - Truncate long proofs/code blocks
      - Merge multiple mentions of the same lemma

    Args:
        messages: Either a pre-rendered string or list of message dicts
            with keys: source, content, msg_type.
        budget: Maximum output characters.
    """
    if isinstance(messages, str):
        return _compress_text_broadcast(messages, budget)

    if not messages:
        return ""

    # Group by type
    positive = []
    negative = []
    partial = []

    for msg in messages:
        mt = msg.get('msg_type', '')
        content = msg.get('content', '')
        source = msg.get('source', '?')

        if 'positive' in mt or 'lemma' in mt or 'discovery' in mt:
            positive.append((source, content))
        elif 'negative' in mt:
            negative.append((source, content))
        elif 'partial' in mt:
            partial.append((source, content))
        else:
            positive.append((source, content))

    parts = []
    used = 0

    # Partial proofs first (highest value)
    if partial:
        parts.append("## Partial progress from teammates")
        used += 40
        for src, content in partial[:3]:
            line = f"- [{src}] {_truncate_line(content, 200)}"
            if used + len(line) + 1 > budget:
                break
            parts.append(line)
            used += len(line) + 1
        parts.append("")

    # Positive discoveries
    if positive:
        # Deduplicate lemma mentions
        seen_lemmas = set()
        deduped = []
        for src, content in positive:
            lemma_match = re.search(r'([A-Z][a-zA-Z]*(?:\.[a-zA-Z_]\w*)+)', content)
            if lemma_match:
                lemma = lemma_match.group(1)
                if lemma in seen_lemmas:
                    continue
                seen_lemmas.add(lemma)
            deduped.append((src, content))

        if deduped:
            parts.append("## Useful discoveries")
            used += 25
            for src, content in deduped[:8]:
                line = f"- [{src}] {_truncate_line(content, 150)}"
                if used + len(line) + 1 > budget:
                    break
                parts.append(line)
                used += len(line) + 1
            parts.append("")

    # Negative knowledge (lower priority, less budget)
    remaining = budget - used
    if negative and remaining > 100:
        parts.append("## Known dead ends (avoid these)")
        used += 35
        for src, content in negative[:5]:
            line = f"- [{src}] {_truncate_line(content, 120)}"
            if used + len(line) + 1 > budget:
                break
            parts.append(line)
            used += len(line) + 1

    return "\n".join(parts)


def _compress_text_broadcast(text: str, budget: int) -> str:
    """Compress a pre-rendered broadcast text string."""
    if len(text) <= budget:
        return text

    lines = text.split('\n')
    # Keep header and deduplicate content lines
    result_lines = []
    seen = set()
    used = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Headers always kept
        if stripped.startswith('#'):
            result_lines.append(line)
            used += len(line) + 1
            continue
        # Deduplicate content
        key = re.sub(r'\[.*?\]', '', stripped)[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        truncated = _truncate_line(stripped, 150)
        if used + len(truncated) + 1 > budget:
            break
        result_lines.append(truncated)
        used += len(truncated) + 1

    return "\n".join(result_lines)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════

def compress_for_prompt(
    text: str,
    budget: int = 1200,
    context_type: str = "general",
) -> str:
    """Generic compression dispatcher.

    Args:
        text: Raw text to compress.
        budget: Max output chars.
        context_type: "error", "feedback", "broadcast", or "general".
    """
    if not text or len(text) <= budget:
        return text

    if context_type == "error":
        return compress_lean_errors(text, budget=budget)
    elif context_type == "feedback":
        return compress_feedback(text, budget=budget)
    elif context_type == "broadcast":
        return _compress_text_broadcast(text, budget=budget)
    else:
        # General text: dedup lines, truncate
        return _compress_text_broadcast(text, budget=budget)

"""common/constants.py — Cross-layer constants.

Single source of truth for values that previously had to be kept in
sync across multiple files. Touch one place, get one effect.
"""
from __future__ import annotations

import os

# ─── LLM defaults ──────────────────────────────────────────────────
#
# The default Claude model used when callers don't override. This used
# to be hard-coded in 8+ places (run_eval.py, run_unified.py x2,
# async_llm_provider.py x3, profiles.py x2). Now everyone reads from
# here, and the env var ``AI4MATH_DEFAULT_MODEL`` lets ops change it
# without a code patch.
DEFAULT_CLAUDE_MODEL: str = os.environ.get(
    "AI4MATH_DEFAULT_MODEL", "claude-sonnet-4-20250514")

# Anthropic API key env var name (one place to change if the convention shifts)
ANTHROPIC_API_KEY_ENV: str = "ANTHROPIC_API_KEY"

"""tests/conftest.py — 共享测试配置

主要职责:
  1. sys.path 注入 (让 ``import agent.* / engine.* / prover.*`` 在
     裸 pytest 调用下也能 work — 没有这个 PYTHONPATH 设置必须靠
     CI / Makefile 注入)。
  2. v15: live-marker gating — ``@pytest.mark.live`` 标记的测试需要
     真实 LLM API key (``ANTHROPIC_API_KEY`` / ``DEEPSEEK_API_KEY``
     / ``OPENAI_API_KEY`` 任一)。在 CI 没 secret 的情况下这些测试
     必须 silently skip 而不是 fail —— 否则 PR 流水线会卡。
  3. v15: lean-marker gating — ``@pytest.mark.lean`` 需要工作的 Lean 4
     工具链 + Mathlib build。同样默认 skip。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── live / lean markers — opt-in only ────────────────────────────────

_LIVE_API_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    # vLLM/sglang/ollama use a base URL not a key; honour those too.
    "OPENAI_API_BASE",
)


def _has_live_api_credentials() -> bool:
    """At least one provider's credential / endpoint is configured."""
    return any(os.environ.get(v) for v in _LIVE_API_ENV_VARS)


def _has_lean_toolchain() -> bool:
    """Heuristic: a ``lean`` binary on PATH that responds to ``--version``.

    Cheap shutil.which check; we don't actually invoke Lean here because
    the conftest runs once per session and a slow probe would tax even
    test collection."""
    import shutil
    return shutil.which("lean") is not None


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``@pytest.mark.live`` / ``@pytest.mark.lean`` when the
    environment can't honour them.

    Users can force-run by passing ``-m live`` or ``-m lean`` AND having
    the relevant credentials/toolchain in place."""
    skip_live = pytest.mark.skip(
        reason="live test skipped: no LLM API credentials in env "
               "(set ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY "
               "or OPENAI_API_BASE for a local vLLM server)")
    skip_lean = pytest.mark.skip(
        reason="lean test skipped: no 'lean' binary on PATH")

    has_live = _has_live_api_credentials()
    has_lean = _has_lean_toolchain()

    for item in items:
        if "live" in item.keywords and not has_live:
            item.add_marker(skip_live)
        if "lean" in item.keywords and not has_lean:
            item.add_marker(skip_lean)

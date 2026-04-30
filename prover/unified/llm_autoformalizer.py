"""prover/unified/llm_autoformalizer.py — Real LLM-based NL→FL translation.

The default ``NLExistenceBridgeTool`` ships a 5-pattern regex
heuristic — useful when nothing better is around, but plainly worse
than any LLM that already lives in the agent loop. This module wraps
an :class:`LLMProvider` as a real autoformalizer that

  * recognises that the agent already has an LLM with sufficient
    coding ability (Claude / GPT-class models),
  * spends ~one extra LLM call per ``nl_existence`` invocation,
  * never claims a confidence the heuristic can't deliver — when
    the LLM call fails, the heuristic still runs as final fallback.

The module is intentionally additive: it does not modify the existing
heuristic path. A user who wants the LLM autoformalizer registers it
explicitly::

    from prover.unified.llm_autoformalizer import (
        register_llm_autoformalizer)
    register_llm_autoformalizer(my_llm_provider)

After that call, every subsequent ``NLExistenceBridgeTool`` invocation
prefers the LLM. To revert::

    from prover.unified.tools_infra import register_autoformalizer
    register_autoformalizer(None)

Public API
----------

* :func:`make_llm_autoformalizer` — build the callable expected by
  :func:`prover.unified.tools_infra.register_autoformalizer`.
* :func:`register_llm_autoformalizer` — convenience wrapper that
  builds and registers in one call.
* :data:`DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT` — the prompt used; can
  be replaced by passing ``system_prompt=`` to the factory.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT = """\
You are a Lean 4 autoformalizer. Translate a natural-language
math problem (and its expected answer type) into a single Lean 4
``theorem`` statement of *existence shape*. Output rules:

1. Output the ``theorem`` declaration only — no Markdown fence,
   no commentary, no ``import`` lines.
2. Use ``theorem ai4math_q : ∃ ...`` as the prefix. The body must
   contain at least one existential ``∃`` over the answer type.
3. Prefer Mathlib-standard symbols: ``ℕ`` ``ℤ`` ``ℚ`` ``ℝ``
   ``Set`` ``Finset``. Do NOT use Greek-letter aliases.
4. If the problem expresses a constraint you cannot encode
   precisely, encode the *shape* of the assertion and leave a
   ``/- TODO: predicate -/`` comment for the human reader.
5. If the answer type is unclear, default to ``ℕ``.

Example input:
  Problem: "Find the smallest natural number n such that n^2 > 100."
  Answer type: integer

Example output:
  theorem ai4math_q : ∃ (n : ℕ),
      n ^ 2 > 100 ∧ ∀ (m : ℕ), m ^ 2 > 100 → n ≤ m
"""


def make_llm_autoformalizer(
        llm: Any,
        *,
        system_prompt: str = DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_passthrough: bool = False,
) -> Callable[[str, str], str]:
    """Build a sync NL→Lean autoformalizer callable.

    Args:
      llm: An object exposing ``generate(system, user, temperature,
           tools, max_tokens)`` returning an :class:`LLMResponse`-shaped
           value (anything with a ``.content`` attribute or a ``content``
           dict key works).
      system_prompt: System prompt sent on every call. Default is
           :data:`DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT`.
      temperature: Generation temperature. Default 0 to maximise
           reproducibility — autoformalization is not a place for
           sampling diversity.
      max_tokens: Maximum tokens to generate. The expected output is
           well under 800 tokens.
      timeout_passthrough: If True, ``RuntimeError`` raised by the LLM
           is re-raised so the caller's heuristic fallback fires.
           If False (default), the error is logged and re-raised so
           the registered-autoformalizer machinery in
           :file:`tools_infra.py` catches it and falls through to
           the heuristic.

    Returns:
      A callable ``(nl_problem: str, answer_type: str) -> str``. The
      string is the LLM's translation, post-processed to strip code
      fences and trim leading whitespace. Empty / mal-formed
      responses raise ``RuntimeError`` so the heuristic fallback runs.

    The returned callable is synchronous; see
    :func:`make_llm_autoformalizer_async` for an async variant.
    """
    if llm is None:
        raise ValueError("make_llm_autoformalizer: llm must not be None")
    if not hasattr(llm, "generate"):
        raise TypeError(
            f"llm must expose .generate(...) — got {type(llm).__name__}")

    def _format_user(nl: str, ans_type: str) -> str:
        return (
            f"Problem (natural language):\n{nl.strip()}\n\n"
            f"Answer type: {ans_type or 'unspecified'}\n\n"
            f"Output the Lean 4 theorem statement now."
        )

    def _autoformalize(nl: str, ans_type: str) -> str:
        if not nl or not nl.strip():
            raise RuntimeError("autoformalize: empty NL problem")
        user = _format_user(nl, ans_type)
        try:
            resp = llm.generate(
                system=system_prompt, user=user,
                temperature=temperature,
                tools=None, max_tokens=max_tokens)
        except Exception as e:
            if timeout_passthrough:
                raise
            logger.debug(f"LLM autoformalizer error: {e}")
            raise RuntimeError(f"LLM autoformalizer failed: {e}") from e

        content = _extract_content(resp)
        cleaned = _strip_lean_fence(content).strip()
        if not cleaned:
            raise RuntimeError(
                "LLM autoformalizer returned empty content")
        if "theorem" not in cleaned:
            # Strict: a non-theorem output is unusable as an
            # autoformalization, raise so heuristic runs.
            raise RuntimeError(
                "LLM autoformalizer output has no theorem keyword")
        return cleaned

    return _autoformalize


def make_llm_autoformalizer_async(
        llm: Any,
        *,
        system_prompt: str = DEFAULT_AUTOFORMALIZER_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 800,
):
    """Async version of :func:`make_llm_autoformalizer`. The returned
    callable is awaited by ``NLExistenceBridgeTool`` in async contexts.
    """
    sync_fn = make_llm_autoformalizer(
        llm, system_prompt=system_prompt,
        temperature=temperature, max_tokens=max_tokens)

    async def _async_translate(nl: str, ans_type: str) -> str:
        # If the underlying generate() is async, prefer that to keep
        # the agent event loop unblocked. Otherwise fall through to
        # the sync path; LLM providers in this project are all sync,
        # so this is overwhelmingly the case.
        agen = getattr(llm, "agenerate", None)
        if agen is None:
            return sync_fn(nl, ans_type)
        try:
            user = (
                f"Problem (natural language):\n{nl.strip()}\n\n"
                f"Answer type: {ans_type or 'unspecified'}\n\n"
                f"Output the Lean 4 theorem statement now.")
            resp = await agen(
                system=system_prompt, user=user,
                temperature=temperature, max_tokens=max_tokens)
            content = _extract_content(resp)
            cleaned = _strip_lean_fence(content).strip()
            if not cleaned or "theorem" not in cleaned:
                raise RuntimeError("empty/invalid async LLM response")
            return cleaned
        except Exception as e:
            logger.debug(f"async LLM autoformalizer error: {e}")
            raise RuntimeError(
                f"async LLM autoformalizer failed: {e}") from e

    return _async_translate


def register_llm_autoformalizer(
        llm: Any, **kwargs) -> Callable[[str, str], str]:
    """Build and register an LLM-based autoformalizer in one call.

    Equivalent to::

        fn = make_llm_autoformalizer(llm, **kwargs)
        register_autoformalizer(fn)

    Returns the callable that was registered, so the caller can
    inspect it / re-register it.
    """
    from prover.unified.tools_infra import register_autoformalizer
    fn = make_llm_autoformalizer(llm, **kwargs)
    register_autoformalizer(fn)
    return fn


# ─── Internals ───────────────────────────────────────────────────────


_LEAN_FENCE_RE = re.compile(
    r"```(?:lean(?:4)?)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_content(resp: Any) -> str:
    """Pull a string out of an LLMResponse or a dict-shaped response.

    Handles three shapes:

    * an object with a ``.content`` attribute (the project's
      :class:`LLMResponse`)
    * a dict with a ``content`` key
    * a plain string (some test mocks)
    """
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        c = resp.content
        return c if isinstance(c, str) else str(c or "")
    if isinstance(resp, dict):
        c = resp.get("content", "")
        return c if isinstance(c, str) else str(c or "")
    return ""


def _strip_lean_fence(text: str) -> str:
    """If the LLM wrapped its output in ```lean fences, peel them off.

    Otherwise return ``text`` unchanged. Multi-block outputs return
    the first fenced block (autoformalization is supposed to be a
    single theorem).
    """
    if not text:
        return ""
    m = _LEAN_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()

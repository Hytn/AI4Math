"""tests/test_smoke_v12.py — Pin v12 bug fixes against regression.

Each test in this module corresponds to one of the latent bugs found
in v11 that was fixed in v12. If any of these starts failing again,
that means we've reintroduced a bug we already paid to fix.

Run: pytest tests/test_smoke_v12.py -v

These are mock-only tests (no Lean, no real LLM). They validate that
the *interfaces* match — which is the class of bug v11 had repeatedly
("接口变了但调用方没跟上"). The fix: every wired-together pair now
has a test pinning the contract.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────
# A.1 — premise_search.py TF-IDF fallback class name + API
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_premise_search_tfidf_fallback_loads_real_lemmas():
    """Verifies the v11→v12 fix to PremiseSearchTool._get_tfidf.

    The previous code imported a non-existent class TFIDFRetriever
    (real name: KnowledgeTFIDFRetriever), passed wrong constructor
    args, and called a non-existent .retrieve() method. Every call
    was silently swallowed by ``except Exception: logger.debug``.
    This test pins each of those contracts.
    """
    from agent.tools.builtin.premise_search import PremiseSearchTool
    from agent.tools.base import ToolContext

    tool = PremiseSearchTool()  # no knowledge_store → must use TF-IDF fallback
    res = await tool.execute({"query": "add_comm", "max_results": 3},
                              ToolContext())
    assert not res.is_error, f"premise_search errored: {res.content}"
    parsed = json.loads(res.content)
    assert isinstance(parsed, list)
    # Heuristic-only mode would return at most a few synthetic entries
    # named like 'add_comm' / 'add_assoc'. The real TF-IDF path returns
    # qualified Mathlib names like 'Nat.add_comm', 'Real.add_comm'.
    sources = {r["source"] for r in parsed}
    assert "tfidf" in sources, (
        "Expected at least one TF-IDF result; got sources={} (this "
        "regression is the v11 latent bug — the TF-IDF fallback path "
        "is silently broken)".format(sources))


# ─────────────────────────────────────────────────────────────────
# A.2 / A.3 — DecomposeSubgoalTool needs LLM, no fake 'kind' field
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decompose_tool_requires_llm_with_clear_error():
    """Without LLM, the tool must say so — not silently fail."""
    from prover.unified.tools_extra import DecomposeSubgoalTool
    from agent.tools.base import ToolContext

    tool = DecomposeSubgoalTool()  # no LLM
    res = await tool.execute({"goal": "n + 0 = n"}, ToolContext())
    assert res.is_error
    assert "no LLM available" in res.content


@pytest.mark.asyncio
async def test_decompose_tool_with_llm_returns_real_subgoal_fields():
    """With LLM, the tool returns real SubGoal fields (name/statement/
    difficulty), not the previous-version fake 'kind' field."""
    from prover.unified.tools_extra import DecomposeSubgoalTool
    from agent.brain.async_llm_provider import LLMResponse
    from agent.tools.base import ToolContext

    class _MockLLM:
        def generate(self, system, user, temperature=0.7):
            return LLMResponse(
                content=("lemma sg1 (n : Nat) : n + 0 = n := by sorry\n"
                          "lemma sg2 (n m : Nat) : n + m = m + n := by sorry"),
                model="mock", tokens_in=10, tokens_out=10, latency_ms=1)

    tool = DecomposeSubgoalTool(llm=_MockLLM())
    res = await tool.execute({"goal": "n + m = m + n"}, ToolContext())
    assert not res.is_error
    parsed = json.loads(res.content)
    assert len(parsed) >= 1
    for sg in parsed:
        # Real fields
        assert "name" in sg
        assert "statement" in sg
        assert "difficulty" in sg
        # Old fake field gone
        assert "kind" not in sg, (
            "Tool re-introduced the fake 'kind' field; v11 returned "
            "this from `getattr(sg, 'kind', 'subgoal')` against a "
            "SubGoal dataclass that has no `kind` attribute.")


# ─────────────────────────────────────────────────────────────────
# C.1 — Model string is one constant
# ─────────────────────────────────────────────────────────────────

def test_default_claude_model_is_centralized():
    from common.constants import DEFAULT_CLAUDE_MODEL
    assert isinstance(DEFAULT_CLAUDE_MODEL, str)
    assert DEFAULT_CLAUDE_MODEL  # non-empty


def test_no_hardcoded_model_strings_outside_constants():
    """The model name should appear only in common/constants.py.
    Hardcoding it elsewhere is the v11-and-earlier mistake of having
    7 places to change when the model rolls."""
    import re
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = re.compile(r'"claude-sonnet-4-\d{8}"')
    offenders = []
    for root, dirs, files in os.walk(here):
        if any(x in root for x in ("/.git", "/__pycache__", "/data", "/tests")):
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            # constants.py is the one allowed home
            if p.endswith("common/constants.py"):
                continue
            with open(p) as fh:
                for n, line in enumerate(fh, 1):
                    if pattern.search(line):
                        offenders.append(f"{p}:{n}")
    assert not offenders, (
        "Hardcoded model strings outside common/constants.py: "
        + "; ".join(offenders))


# ─────────────────────────────────────────────────────────────────
# C.2 — AsyncCachedProvider caches both generate() and chat()
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_provider_caches_chat():
    """v11: AsyncCachedProvider only overrode generate(). AgentLoop calls
    chat() preferentially, so the cache was bypassed in production.
    v12 adds a chat() override; this test pins it."""
    from agent.brain.async_llm_provider import (
        AsyncMockProvider, AsyncCachedProvider)

    inner = AsyncMockProvider()
    cached = AsyncCachedProvider(inner, cache_all=True)
    msgs = [{"role": "user", "content": "hello"}]

    r1 = await cached.chat(system="S", messages=msgs, temperature=0.7)
    r2 = await cached.chat(system="S", messages=msgs, temperature=0.7)

    assert not r1.cached
    assert r2.cached, "Identical chat() call should be a cache hit"
    assert cached.hits >= 1


@pytest.mark.asyncio
async def test_cache_provider_caches_generate():
    """Sanity check that the original generate() cache still works
    after the chat() override."""
    from agent.brain.async_llm_provider import (
        AsyncMockProvider, AsyncCachedProvider)

    cached = AsyncCachedProvider(AsyncMockProvider(), cache_all=False)
    # cache_all=False: only T=0 is cached
    a = await cached.generate(system="S", user="hi", temperature=0.0)
    b = await cached.generate(system="S", user="hi", temperature=0.0)
    assert b.cached
    # T != 0 → not cached
    c = await cached.generate(system="S", user="hi", temperature=0.7)
    d = await cached.generate(system="S", user="hi", temperature=0.7)
    assert not c.cached and not d.cached


# ─────────────────────────────────────────────────────────────────
# C.3 — integrity_strict toggle works end-to-end
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_integrity_strict_toggle():
    """``native_decide`` must pass in non-strict mode (Mathlib usage),
    fail in strict mode (competition usage); ``sorry`` must always
    fail regardless of strictness."""
    from agent.tools.builtin.lean_verify import LeanVerifyTool
    from agent.tools.base import ToolContext
    from engine._core import FullVerifyResult

    class _MockPool:
        base_env_id = 0
        async def verify_complete(self, statement, proof, preamble):
            return FullVerifyResult(
                success=True, has_sorry=False, errors=[], stderr="",
                goals_remaining=[], elapsed_ms=10)

    code_native = "theorem t : 1 + 1 = 2 := by native_decide"
    code_sorry  = "theorem t : 1 + 1 = 2 := by sorry"

    relaxed = LeanVerifyTool(lean_pool=_MockPool(), integrity_strict=False)
    strict  = LeanVerifyTool(lean_pool=_MockPool(), integrity_strict=True)

    r1 = json.loads((await relaxed.execute({"code": code_native},
                                            ToolContext())).content)
    r2 = json.loads((await strict.execute({"code": code_native},
                                            ToolContext())).content)
    r3 = json.loads((await relaxed.execute({"code": code_sorry},
                                            ToolContext())).content)

    assert r1["verified"] is True
    assert r2["verified"] is False
    assert r3["verified"] is False  # sorry rejected in any mode
    # Relaxed mode must surface the violation as advisory
    assert "integrity_violations" in r1
    assert "integrity_note" in r1


# ─────────────────────────────────────────────────────────────────
# B — dead modules are deleted, dead fields are gone
# ─────────────────────────────────────────────────────────────────

def test_dead_common_modules_deleted():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ("common/hook_types.py", "common/budget.py",
              "common/working_memory.py"):
        assert not os.path.exists(os.path.join(here, f)), \
            f"{f} should have been deleted in v12"


def test_dead_codegen_dir_deleted():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert not os.path.exists(os.path.join(here, "prover/codegen")), \
        "prover/codegen/ was empty in v11; should be deleted in v12"


def test_dead_observation_fields_gone():
    """compress_errors_budget and visible_history_turns had 0 readers."""
    from prover.unified.profiles import ObservationPolicy
    op = ObservationPolicy()
    assert not hasattr(op, "compress_errors_budget")
    assert not hasattr(op, "visible_history_turns")


def test_dead_profile_plugins_field_gone():
    from prover.unified.profiles import Profile
    p = Profile(name="t")
    assert not hasattr(p, "plugins")


def test_yaml_with_legacy_fields_still_loads():
    """v12 keeps a compat shim so old YAMLs (with plugins / dead obs
    fields) still load. This pins that contract."""
    import tempfile
    from prover.unified import load_profile_from_yaml

    legacy = """
name: legacy_test
description: pretends to be a v10 YAML
tools: [lean_verify]
max_turns: 4
framing: whole_proof_repair
observation:
  auto_inject_lean_compile: true
  compress_errors_budget: 1200
  visible_history_turns: -1
  inject_premises_in_prompt: true
plugins: []
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(legacy)
        path = f.name
    try:
        prof = load_profile_from_yaml(path)
        assert prof.name == "legacy_test"
        assert prof.max_turns == 4
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────────
# A.5 — check_layers passes
# ─────────────────────────────────────────────────────────────────

def test_check_layers_passes():
    """engine/ must not import from agent/ or prover/."""
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "scripts/check_layers.py"],
        cwd=here, capture_output=True, text=True)
    assert result.returncode == 0, (
        "check_layers.py failed:\n" + result.stdout + result.stderr)


# ─────────────────────────────────────────────────────────────────
# Smoke: PRESETS still loads, all 14 profiles round-trip
# ─────────────────────────────────────────────────────────────────

def test_all_presets_load():
    from prover.unified.profiles import PRESETS, get_profile
    assert len(PRESETS) >= 14
    for name in PRESETS:
        p = get_profile(name)
        assert p.name == name


def test_all_yaml_profiles_round_trip():
    """Every YAML in config/profiles must load without error and
    produce a Profile instance. Pins the YAML schema."""
    import glob
    from prover.unified import load_profile_from_yaml

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_files = sorted(glob.glob(os.path.join(here, "config/profiles/*.yaml")))
    assert yaml_files, "no YAML profiles found"
    for f in yaml_files:
        prof = load_profile_from_yaml(f)
        assert prof.name, f"profile {f} has empty name"

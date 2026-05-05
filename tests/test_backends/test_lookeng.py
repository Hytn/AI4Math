"""tests/test_backends/test_lookeng.py — LooKeng stateless-REPL tests.

These tests use a custom mock REPLTransport that records each ``cmd``
it receives, so we can confirm LooKeng correctly composes the running
context block before sending it to the inner transport.
"""
import pytest

from engine.backends.lookeng import (
    LooKengBackend, RunningContext, LemmaCacheEntry,
    build_running_context_prompt,
)
from engine.transport import REPLTransport

# ─── A scriptable inner transport ───────────────────────────

class ScriptedInner(REPLTransport):
    """Records all commands it sees and replies according to a script.

    ``script`` is a list of (predicate, response) tuples. The first
    predicate that returns truthy on a command provides the response.
    """

    def __init__(self, script=None):
        self.calls = []
        self._alive = False
        self._script = list(script or [])

    async def start(self):
        self._alive = True
        return True

    async def send(self, cmd):
        self.calls.append(cmd)
        for pred, resp in self._script:
            if pred(cmd):
                return resp
        # default success
        return {"env": 1, "messages": [], "goals": []}

    async def close(self):
        self._alive = False

    @property
    def is_alive(self):
        return self._alive

# ─── LemmaCacheEntry ─────────────────────────────────────────

def test_lemma_cache_render_for_llm_strips_whitespace():
    e = LemmaCacheEntry(name="lem1", statement="  theorem lem1 : True   ")
    assert e.render_for_llm() == "theorem lem1 : True"

def test_lemma_cache_render_for_compiler_with_proof():
    e = LemmaCacheEntry(
        name="lem1",
        statement="theorem lem1 : 1 + 1 = 2",
        proof_body="rfl")
    out = e.render_for_compiler()
    assert "theorem lem1 : 1 + 1 = 2" in out
    assert ":= by" in out
    assert "rfl" in out

def test_lemma_cache_render_for_compiler_falls_back_to_sorry():
    e = LemmaCacheEntry(name="lem1", statement="theorem lem1 : True")
    out = e.render_for_compiler()
    assert "sorry" in out

def test_lemma_cache_render_for_compiler_handles_term_mode_proof():
    """Proof bodies starting with `:=` are term-mode, not tactic-mode."""
    e = LemmaCacheEntry(
        name="lem1",
        statement="theorem lem1 : 1 = 1",
        proof_body=":= rfl")
    out = e.render_for_compiler()
    assert ":= rfl" in out
    # Should not double-add `:= by`
    assert ":= by" not in out

# ─── RunningContext ──────────────────────────────────────────

def test_running_context_append_and_render_briefing():
    ctx = RunningContext(
        theorem_header="theorem main : True",
        preamble="import Mathlib")
    assert ctx.render_llm_briefing() == ""

    ctx.append(LemmaCacheEntry(
        name="step1",
        statement="theorem step1 : 1 + 1 = 2",
        proof_body="rfl"))
    ctx.append(LemmaCacheEntry(
        name="step2",
        statement="theorem step2 : 2 + 2 = 4",
        proof_body="rfl"))

    briefing = ctx.render_llm_briefing()
    assert "step1" in briefing
    assert "step2" in briefing
    assert "1 + 1 = 2" in briefing
    # proof bodies should NOT leak into the LLM briefing
    assert "rfl" not in briefing

def test_running_context_render_compiler_block_includes_everything():
    ctx = RunningContext(theorem_header="theorem main : True",
                          preamble="import Mathlib")
    ctx.append(LemmaCacheEntry(
        name="step1",
        statement="theorem step1 : 1 + 1 = 2",
        proof_body="rfl"))
    block = ctx.render_compiler_block("theorem main : True := trivial")
    assert "import Mathlib" in block
    assert "step1" in block
    assert "rfl" in block  # proof body IS in the compiler block
    assert "trivial" in block

def test_running_context_fork_is_independent():
    ctx = RunningContext(theorem_header="T", preamble="import Mathlib")
    ctx.append(LemmaCacheEntry(name="a", statement="theorem a : True",
                                 proof_body="trivial"))
    fork = ctx.fork(session_id="branch-1")
    fork.append(LemmaCacheEntry(name="b", statement="theorem b : True",
                                  proof_body="trivial"))
    assert len(ctx.lemmas) == 1
    assert len(fork.lemmas) == 2
    assert fork.session_id == "branch-1"

# ─── LooKengBackend session lifecycle ────────────────────────

@pytest.mark.asyncio
async def test_begin_session_creates_running_context():
    inner = ScriptedInner()
    lk = LooKengBackend(inner=inner)
    await lk.start()

    sid = await lk.begin_session(
        theorem="theorem main : 1 + 1 = 2",
        preamble="import Mathlib")
    assert sid
    ctx = lk.get_running_context(sid)
    assert ctx is not None
    assert ctx.theorem_header == "theorem main : 1 + 1 = 2"
    assert ctx.preamble == "import Mathlib"
    assert ctx.lemmas == []

@pytest.mark.asyncio
async def test_submit_lemma_success_appends_to_context():
    """A successful compile should grow the running context."""
    inner = ScriptedInner(script=[
        # Always return success
        (lambda c: True, {"env": 1, "messages": [], "goals": []}),
    ])
    lk = LooKengBackend(inner=inner)
    await lk.start()
    sid = await lk.begin_session(theorem="theorem main : True")
    r = await lk.submit_lemma(
        session_id=sid,
        name="step1",
        statement="theorem step1 : 1 + 1 = 2",
        proof="rfl")
    assert r["ok"] is True
    assert r["running_context_size"] == 1

@pytest.mark.asyncio
async def test_submit_lemma_failure_doesnt_mutate_context():
    """A type-check failure must leave the running context alone."""
    inner = ScriptedInner(script=[
        (lambda c: True,
         {"env": 1,
          "messages": [{"severity": "error", "data": "type mismatch"}],
          "goals": []}),
    ])
    lk = LooKengBackend(inner=inner)
    await lk.start()
    sid = await lk.begin_session(theorem="theorem main : True")
    r = await lk.submit_lemma(
        session_id=sid, name="bad", statement="theorem bad : False",
        proof="trivial")
    assert r["ok"] is False
    assert r["running_context_size"] == 0
    assert "type mismatch" in r["errors"][0]

@pytest.mark.asyncio
async def test_submit_lemma_unknown_session_errors():
    inner = ScriptedInner()
    lk = LooKengBackend(inner=inner)
    await lk.start()
    r = await lk.submit_lemma(
        session_id="nope", name="x",
        statement="theorem x : True", proof="trivial")
    assert r["ok"] is False
    assert "unknown session" in r["errors"][0]

@pytest.mark.asyncio
async def test_submit_lemma_max_size_guard():
    """Hitting max_running_context_lemmas must refuse, not silently truncate."""
    inner = ScriptedInner()
    lk = LooKengBackend(inner=inner, max_running_context_lemmas=2)
    await lk.start()
    sid = await lk.begin_session(theorem="theorem main : True")
    # Fill to capacity
    for i in range(2):
        await lk.submit_lemma(
            session_id=sid, name=f"l{i}",
            statement=f"theorem l{i} : True", proof="trivial")
    r = await lk.submit_lemma(
        session_id=sid, name="overflow",
        statement="theorem overflow : True", proof="trivial")
    assert r["ok"] is False
    assert "max" in r["errors"][0].lower()

@pytest.mark.asyncio
async def test_close_session_drops_context():
    inner = ScriptedInner()
    lk = LooKengBackend(inner=inner)
    await lk.start()
    sid = await lk.begin_session(theorem="theorem main : True")
    assert lk.get_running_context(sid) is not None
    resp = await lk.send({"lookeng_op": "close_session", "session_id": sid})
    assert resp["ok"] is True
    assert lk.get_running_context(sid) is None

@pytest.mark.asyncio
async def test_compiler_block_assembled_correctly():
    """Verify the full snippet handed to the inner transport contains
    preamble + previous lemmas + new proof in that order."""
    inner = ScriptedInner()
    lk = LooKengBackend(inner=inner)
    await lk.start()
    sid = await lk.begin_session(
        theorem="theorem main : True", preamble="import Mathlib")
    await lk.submit_lemma(
        session_id=sid, name="a",
        statement="theorem a : 1 = 1", proof="rfl")
    # The second submit should compile a snippet that includes lemma `a`.
    await lk.submit_lemma(
        session_id=sid, name="b",
        statement="theorem b : 2 = 2", proof="rfl")
    second_compile = inner.calls[1]["cmd"]
    assert "import Mathlib" in second_compile
    assert "theorem a : 1 = 1" in second_compile
    assert "theorem b : 2 = 2" in second_compile
    # Order: preamble, then a, then b
    p_idx = second_compile.index("import Mathlib")
    a_idx = second_compile.index("theorem a")
    b_idx = second_compile.index("theorem b")
    assert p_idx < a_idx < b_idx

# ─── build_running_context_prompt ────────────────────────────

def test_running_context_prompt_omits_preamble_by_default():
    ctx = RunningContext(
        theorem_header="theorem main : True", preamble="import Mathlib")
    out = build_running_context_prompt(ctx)
    assert "import Mathlib" not in out
    assert "theorem main" in out

def test_running_context_prompt_includes_preamble_when_asked():
    ctx = RunningContext(
        theorem_header="theorem main : True", preamble="import Mathlib")
    out = build_running_context_prompt(ctx, include_preamble=True)
    assert "import Mathlib" in out

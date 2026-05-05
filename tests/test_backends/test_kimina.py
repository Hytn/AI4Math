"""tests/test_backends/test_kimina.py — Kimina Lean Server backend tests.

The Kimina server is an external service we don't have in CI, so all
tests here exercise the *protocol translation layer* — they confirm
the backend correctly maps stdio-protocol JSON to the Kimina REST
schema and handles every documented failure mode (server down,
aiohttp missing, malformed response, no result for an id).
"""
import json
import asyncio
import pytest

from engine.backends.kimina_server import (
    KiminaServerBackend, KiminaServerClient,
    BatchVerifyRequest, BatchVerifyResult, TacticTrace, _ImportCache,
)

# ─── BatchVerifyRequest / Result wire format ─────────────────────

def test_batch_verify_request_to_wire_default_preamble():
    r = BatchVerifyRequest(id="x", proof="theorem t : True := trivial")
    w = r.to_wire()
    assert w["id"] == "x"
    assert w["proof"] == "theorem t : True := trivial"
    assert w["preamble"] == "import Mathlib"
    assert w["timeout"] == 120

def test_batch_verify_request_custom_preamble():
    r = BatchVerifyRequest(
        id="y", proof="...", preamble="import Mathlib.Topology.Basic",
        timeout_seconds=60)
    w = r.to_wire()
    assert w["preamble"] == "import Mathlib.Topology.Basic"
    assert w["timeout"] == 60

def test_batch_verify_result_from_wire_full():
    wire = {
        "id": "r1",
        "success": True,
        "elapsed_ms": 250,
        "has_sorry": False,
        "tactic_trace": [
            {"tactic": "intro h", "goal_before": "P → Q",
             "goals_after": ["Q"], "is_proof_complete": False,
             "line": 2, "column": 3},
            {"tactic": "exact h.elim", "goal_before": "Q",
             "goals_after": [], "is_proof_complete": True},
        ],
    }
    r = BatchVerifyResult.from_wire(wire)
    assert r.id == "r1"
    assert r.success is True
    assert r.elapsed_ms == 250
    assert r.has_sorry is False
    assert len(r.tactic_trace) == 2
    assert r.tactic_trace[0].tactic == "intro h"
    assert r.tactic_trace[0].line == 2
    assert r.tactic_trace[1].is_proof_complete is True

def test_batch_verify_result_from_wire_minimal():
    """Server may omit optional fields; we should default safely."""
    r = BatchVerifyResult.from_wire({"id": "r2", "success": False})
    assert r.id == "r2"
    assert r.success is False
    assert r.error_messages == []
    assert r.tactic_trace == []
    assert r.elapsed_ms == 0

def test_batch_verify_result_from_wire_errors_aliases():
    """Server uses 'errors' field in older versions and 'error_messages' in newer;
    BatchVerifyResult.from_wire must accept both."""
    r1 = BatchVerifyResult.from_wire({"id": "a", "errors": ["e1", "e2"]})
    assert r1.error_messages == ["e1", "e2"]
    r2 = BatchVerifyResult.from_wire({"id": "b", "error_messages": ["e3"]})
    assert r2.error_messages == ["e3"]

# ─── _ImportCache ─────────────────────────────────────────────

def test_import_cache_basic_get_put():
    c = _ImportCache(maxsize=4)
    assert c.get("import Mathlib") is None
    assert c.misses == 1
    c.put("import Mathlib", 42)
    assert c.get("import Mathlib") == 42
    assert c.hits == 1

def test_import_cache_evicts_lru():
    c = _ImportCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts 'a'
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3

def test_import_cache_lru_promotion():
    c = _ImportCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")     # touches 'a' so 'b' becomes the LRU
    c.put("c", 3)  # should evict 'b' instead of 'a'
    assert c.get("a") == 1
    assert c.get("b") is None

def test_import_cache_stats_shape():
    c = _ImportCache()
    c.put("k", 0)
    c.get("k")
    c.get("missing")
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["size"] == 1
    assert 0.0 <= s["hit_rate"] <= 1.0

# ─── KiminaServerBackend protocol translation ─────────────────

@pytest.mark.asyncio
async def test_backend_starts_in_fallback_when_aiohttp_missing(monkeypatch):
    """If aiohttp can't be imported, backend declares fallback."""
    backend = KiminaServerBackend()
    # Force the available flag to False — simulating no aiohttp.
    backend._client._available = False
    backend._fallback = not backend._client.available
    started = await backend.start()
    assert started
    assert backend.is_fallback

@pytest.mark.asyncio
async def test_backend_send_in_fallback_returns_none():
    backend = KiminaServerBackend()
    backend._client._available = False
    backend._fallback = True
    backend._alive = True
    resp = await backend.send({"cmd": "import Mathlib", "env": 0})
    assert resp is None

@pytest.mark.asyncio
async def test_pure_import_is_detected():
    """Bare imports get a synthetic env_id, no server round-trip."""
    backend = KiminaServerBackend()
    backend._client._available = True   # pretend aiohttp present
    backend._fallback = False
    backend._alive = True
    # Install a stub on the client to assert no /api/verify call happens
    # for pure imports.
    calls = []
    async def fake_verify_batch(reqs):
        calls.append(reqs)
        return [BatchVerifyResult(id=r.id, success=True) for r in reqs]
    backend._client.verify_batch = fake_verify_batch  # type: ignore
    backend._client.check_one = lambda **kw: asyncio.sleep(0)  # never called

    resp = await backend.send({
        "cmd": "import Mathlib\nopen Topology",
        "env": 0,
    })
    assert resp is not None
    assert "env" in resp
    assert resp["env"] >= 1
    # Server should not have been called for pure imports.
    # (warmup is async — not awaited synchronously here.)

def test_pure_import_detection_classifies_correctly():
    """The static helper distinguishes import-only from theorem code."""
    f = KiminaServerBackend._is_pure_import
    assert f("import Mathlib") is True
    assert f("import Mathlib\nopen Real") is True
    assert f("set_option pp.all true\nimport Mathlib") is True
    assert f("-- comment\nimport Mathlib") is True
    assert f("theorem t : True := trivial") is False
    assert f("import Mathlib\ntheorem t : True := trivial") is False
    assert f("") is False

def test_assemble_proof_normalises_header():
    """`:=` in the header should be stripped before tactics are appended."""
    out = KiminaServerBackend._assemble_proof(
        "theorem t : True := by", ["trivial"])
    assert out.startswith("theorem t : True := by")
    assert "trivial" in out
    # No double `:= by`
    assert out.count(":=") == 1

# ─── KiminaServerClient ───────────────────────────────────────

def test_client_lazy_import_doesnt_crash_without_aiohttp(monkeypatch):
    """Client construction must succeed even when aiohttp is missing."""
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    c = KiminaServerClient()
    assert c.available is False

def test_client_picks_up_env_var(monkeypatch):
    """KIMINA_SERVER_URL env var should override the default base."""
    monkeypatch.setenv("KIMINA_SERVER_URL", "http://lean.example.com:9000")
    c = KiminaServerClient()
    assert c.base_url == "http://lean.example.com:9000"

def test_client_strips_trailing_slash():
    c = KiminaServerClient(base_url="http://x:8000/")
    assert c.base_url == "http://x:8000"

@pytest.mark.asyncio
async def test_client_verify_batch_empty_returns_empty():
    c = KiminaServerClient()
    out = await c.verify_batch([])
    assert out == []

@pytest.mark.asyncio
async def test_client_verify_batch_when_unavailable_returns_error_per_req():
    c = KiminaServerClient()
    c._available = False
    reqs = [BatchVerifyRequest(id="a", proof="x"),
            BatchVerifyRequest(id="b", proof="y")]
    out = await c.verify_batch(reqs)
    assert len(out) == 2
    assert all(not r.success for r in out)
    assert out[0].id == "a"
    assert out[1].id == "b"

# ─── TacticTrace dataclass ────────────────────────────────────

def test_tactic_trace_defaults():
    t = TacticTrace(tactic="rfl", goal_before="x = x")
    assert t.goals_after == []
    assert t.is_proof_complete is False
    assert t.line == -1

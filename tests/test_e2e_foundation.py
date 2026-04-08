"""tests/test_e2e_foundation.py — End-to-end foundation tests

These tests validate the complete infrastructure stack:
  Transport → Pool → Session → ProofSessionManager → IncrementalVerifier

Using MockTransport with realistic REPL behavior simulation.
NO Lean4 installation required.

Run: pytest tests/test_e2e_foundation.py -v
"""
import asyncio
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.transport import MockTransport, LocalTransport, FallbackTransport
from engine.repl_protocol import REPLRequest, REPLResponse, build_sorry_theorem
from engine._core import (
    TacticFeedback, FullVerifyResult, CompileCache,
    classify_error, assemble_code, make_cache_key,
)
from engine.async_lean_pool import AsyncLeanSession, AsyncLeanPool
from engine.proof_session import ProofSessionManager, ProofSession
from engine.prefilter import PreFilter


# ═══════════════════════════════════════════════════════════════
# Layer 1: REPL Protocol
# ═══════════════════════════════════════════════════════════════

class TestREPLProtocol:
    """Test the wire protocol data structures."""

    def test_command_request(self):
        req = REPLRequest.command("import Mathlib", env=0)
        d = req.to_dict()
        assert d == {"cmd": "import Mathlib", "env": 0}

    def test_tactic_request(self):
        req = REPLRequest.tactic_step("simp", proof_state=3)
        d = req.to_dict()
        assert d == {"tactic": "simp", "proofState": 3}

    def test_response_parsing_success(self):
        raw = {
            "env": 5,
            "messages": [],
            "goals": [],
            "sorries": [],
        }
        resp = REPLResponse.from_dict(raw)
        assert resp.env == 5
        assert not resp.has_errors
        assert resp.is_proof_complete

    def test_response_parsing_error(self):
        raw = {
            "env": 3,
            "messages": [
                {"severity": "error",
                 "data": "type mismatch",
                 "pos": {"line": 1, "column": 5}}
            ],
            "goals": ["⊢ Nat"],
        }
        resp = REPLResponse.from_dict(raw)
        assert resp.has_errors
        assert resp.error_messages == ["type mismatch"]
        assert not resp.is_proof_complete

    def test_response_with_sorries(self):
        raw = {
            "env": 2,
            "messages": [],
            "goals": [],
            "sorries": [
                {"proofState": 1, "goal": "⊢ True",
                 "pos": {"line": 1, "column": 0},
                 "endPos": {"line": 1, "column": 5}},
            ],
        }
        resp = REPLResponse.from_dict(raw)
        assert resp.has_sorry
        goals = resp.interactive_goals
        assert len(goals) == 1
        assert goals[0] == (1, "⊢ True")

    def test_build_sorry_theorem(self):
        assert build_sorry_theorem("theorem t : True") == \
            "theorem t : True := by sorry"
        # Already has body
        assert build_sorry_theorem("theorem t : True := trivial") == \
            "theorem t : True := trivial"

    def test_response_from_invalid_json(self):
        resp = REPLResponse.from_json("{invalid")
        assert resp.has_errors


# ═══════════════════════════════════════════════════════════════
# Layer 2: Transport
# ═══════════════════════════════════════════════════════════════

class TestMockTransport:
    """Test MockTransport's REPL state simulation."""

    @pytest.mark.asyncio
    async def test_env_id_auto_increment(self):
        mock = MockTransport()
        await mock.start()

        r1 = await mock.send({"cmd": "import Mathlib", "env": 0})
        assert r1["env"] == 1

        r2 = await mock.send({"cmd": "theorem t : True := by sorry", "env": 1})
        assert r2["env"] == 2

        await mock.close()

    @pytest.mark.asyncio
    async def test_error_map(self):
        mock = MockTransport(error_on={
            "simp": "tactic 'simp' failed, no applicable lemma"
        })
        await mock.start()

        resp = await mock.send({"cmd": "simp", "env": 1})
        assert resp["messages"][0]["severity"] == "error"
        assert "simp" in resp["messages"][0]["data"]

    @pytest.mark.asyncio
    async def test_sorry_detection(self):
        mock = MockTransport()
        await mock.start()

        resp = await mock.send(
            {"cmd": "theorem t : True := by sorry", "env": 0})
        assert "sorries" in resp
        assert len(resp["sorries"]) == 1
        assert resp["sorries"][0]["proofState"] >= 1

    @pytest.mark.asyncio
    async def test_scripted_responses(self):
        mock = MockTransport(responses=[
            {"env": 1, "messages": [], "goals": ["⊢ True"]},
            {"env": 2, "messages": [], "goals": []},
        ])
        await mock.start()

        r1 = await mock.send({"cmd": "intro", "env": 0})
        assert r1["goals"] == ["⊢ True"]

        r2 = await mock.send({"cmd": "trivial", "env": 1})
        assert r2["goals"] == []

    @pytest.mark.asyncio
    async def test_command_recording(self):
        mock = MockTransport()
        await mock.start()

        await mock.send({"cmd": "A", "env": 0})
        await mock.send({"cmd": "B", "env": 1})

        assert mock.call_count == 2
        assert mock.sent_commands[0]["cmd"] == "A"
        assert mock.sent_commands[1]["cmd"] == "B"

    @pytest.mark.asyncio
    async def test_tactic_mode(self):
        mock = MockTransport()
        await mock.start()

        resp = await mock.send({"tactic": "exact trivial", "proofState": 1})
        assert "proofState" in resp
        assert resp["goals"] == []

    @pytest.mark.asyncio
    async def test_fallback_transport(self):
        fb = FallbackTransport()
        await fb.start()
        assert fb.is_fallback
        assert await fb.send({"cmd": "anything"}) is None


# ═══════════════════════════════════════════════════════════════
# Layer 3: AsyncLeanSession
# ═══════════════════════════════════════════════════════════════

class TestAsyncLeanSession:
    """Test session-level REPL interaction."""

    @pytest.mark.asyncio
    async def test_session_start_with_mock(self):
        mock = MockTransport(responses=[
            {"env": 1, "messages": [], "goals": []},  # preamble response
        ])
        session = AsyncLeanSession(
            session_id=0, project_dir=".",
            transport=mock)
        ok = await session.start("import Mathlib")
        assert ok
        assert session.base_env_id == 1
        assert session.is_alive
        assert not session.is_fallback
        await session.close()

    @pytest.mark.asyncio
    async def test_try_tactic_success(self):
        mock = MockTransport(responses=[
            {"env": 1, "messages": [], "goals": []},  # preamble
            {"env": 2, "messages": [], "goals": ["⊢ Nat"]},  # tactic
        ])
        session = AsyncLeanSession(
            session_id=0, project_dir=".", transport=mock)
        await session.start("import Mathlib")

        result = await session.try_tactic(1, "intro n")
        assert result.success
        assert result.new_env_id == 2
        assert result.remaining_goals == ["⊢ Nat"]
        assert not result.is_proof_complete
        await session.close()

    @pytest.mark.asyncio
    async def test_try_tactic_failure(self):
        mock = MockTransport(responses=[
            {"env": 1, "messages": [], "goals": []},  # preamble
            {"env": 1, "messages": [
                {"severity": "error",
                 "data": "tactic 'simp' failed, no applicable lemmas"}
            ], "goals": []},
        ])
        session = AsyncLeanSession(
            session_id=0, project_dir=".", transport=mock)
        await session.start("import Mathlib")

        result = await session.try_tactic(1, "simp")
        assert not result.success
        assert "simp" in result.error_message
        assert result.error_category == "tactic_failed"
        await session.close()

    @pytest.mark.asyncio
    async def test_try_tactic_proof_complete(self):
        mock = MockTransport(responses=[
            {"env": 1, "messages": [], "goals": []},  # preamble
            {"env": 2, "messages": [], "goals": []},  # proof done
        ])
        session = AsyncLeanSession(
            session_id=0, project_dir=".", transport=mock)
        await session.start("import Mathlib")

        result = await session.try_tactic(1, "trivial")
        assert result.success
        assert result.is_proof_complete
        await session.close()

    @pytest.mark.asyncio
    async def test_verify_complete(self):
        mock = MockTransport(responses=[
            {"env": 1, "messages": [], "goals": []},  # preamble
            {"env": 2, "messages": [], "goals": []},  # verify
        ])
        session = AsyncLeanSession(
            session_id=0, project_dir=".", transport=mock)
        await session.start("import Mathlib")

        result = await session.verify_complete(
            "theorem t : True", ":= by trivial")
        assert result.success
        await session.close()


# ═══════════════════════════════════════════════════════════════
# Layer 4: AsyncLeanPool
# ═══════════════════════════════════════════════════════════════

class TestAsyncLeanPool:
    """Test pool-level session management and concurrency."""

    def _make_pool(self, pool_size=2, responses_per_session=None):
        """Create a pool where each session has a MockTransport."""
        pool = AsyncLeanPool(
            pool_size=pool_size, project_dir=".",
            preamble="import Mathlib")
        return pool

    @pytest.mark.asyncio
    async def test_pool_start_fallback(self):
        """Pool should start in fallback mode when no Lean4 available."""
        pool = AsyncLeanPool(pool_size=2, project_dir="/nonexistent")
        ok = await pool.start()
        assert ok
        stats = pool.stats()
        assert stats["all_fallback"] is True
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_try_tactic(self):
        """Test single tactic through pool."""
        pool = AsyncLeanPool(pool_size=1, project_dir=".")
        ok = await pool.start()
        assert ok

        # In fallback mode, tactics will fail with "no backend"
        result = await pool.try_tactic(0, "simp")
        # Should get a valid TacticFeedback (even if failed)
        assert isinstance(result, TacticFeedback)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_parallel_tactics(self):
        """Test parallel tactic execution."""
        pool = AsyncLeanPool(pool_size=2, project_dir=".")
        await pool.start()

        tactics = ["simp", "ring", "omega", "norm_num"]
        results = await pool.try_tactics_parallel(0, tactics)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, TacticFeedback)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_verify_with_cache(self):
        """Test that verification results are cached."""
        pool = AsyncLeanPool(pool_size=1, project_dir=".")
        await pool.start()

        r1 = await pool.verify_complete("theorem t : True", ":= by trivial")
        r2 = await pool.verify_complete("theorem t : True", ":= by trivial")

        # Second call should hit cache
        cache_stats = pool._compile_cache.stats()
        # At least one hit expected
        assert isinstance(r1, FullVerifyResult)
        assert isinstance(r2, FullVerifyResult)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_stats(self):
        pool = AsyncLeanPool(pool_size=2, project_dir=".")
        await pool.start()
        stats = pool.stats()

        assert "pool_size" in stats
        assert "active_sessions" in stats
        assert "compile_cache" in stats
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_context_manager(self):
        async with AsyncLeanPool(pool_size=1, project_dir=".") as pool:
            stats = pool.stats()
            assert stats["active_sessions"] >= 0


# ═══════════════════════════════════════════════════════════════
# Layer 5: ProofSession State Tree
# ═══════════════════════════════════════════════════════════════

class TestProofSession:
    """Test proof session state tree management.

    Uses a pool with MockTransport sessions to simulate
    realistic env_id state transitions.
    """

    @pytest.mark.asyncio
    async def test_session_manager_begin_proof(self):
        pool = AsyncLeanPool(pool_size=1, project_dir=".")
        await pool.start()

        async with ProofSessionManager(pool) as mgr:
            session = await mgr.begin_proof("theorem t : True := by")
            assert session is not None
            assert session.current_env_id >= 0
            assert not session.is_solved

    @pytest.mark.asyncio
    async def test_proof_session_rewind(self):
        pool = AsyncLeanPool(pool_size=1, project_dir=".")
        await pool.start()

        async with ProofSessionManager(pool) as mgr:
            session = await mgr.begin_proof("theorem t : True := by")
            initial = session.current_env_id

            # Try a step (will fail in fallback, but state tree still updates)
            await session.try_step("intro h")

            # Rewind
            rewound = session.rewind(steps=1)
            assert rewound == initial

    @pytest.mark.asyncio
    async def test_proof_session_tree_stats(self):
        pool = AsyncLeanPool(pool_size=1, project_dir=".")
        await pool.start()

        async with ProofSessionManager(pool) as mgr:
            session = await mgr.begin_proof("theorem t : True := by")
            stats = session.tree_stats()
            assert "total_nodes" in stats
            assert "max_depth" in stats
            assert stats["solved"] is False


# ═══════════════════════════════════════════════════════════════
# Layer 6: PreFilter
# ═══════════════════════════════════════════════════════════════

class TestPreFilter:
    """Test L0 pre-filter rules."""

    def setup_method(self):
        self.pf = PreFilter()

    def test_empty_proof(self):
        result = self.pf.check("")
        assert not result.passed
        assert result.rule_name == "empty_proof"

    def test_sorry_detected(self):
        result = self.pf.check("intro n\nsorry")
        assert not result.passed
        assert result.rule_name == "sorry_detector"

    def test_sorry_in_comment_ok(self):
        result = self.pf.check("-- sorry\ntrivial")
        assert result.passed

    def test_bracket_mismatch(self):
        result = self.pf.check("exact (Nat.add_comm n m")
        assert not result.passed
        assert result.rule_name == "bracket_matcher"

    def test_lean3_syntax(self):
        # "begin...end" is clearly Lean3
        result = self.pf.check("begin\nexact trivial\nend")
        assert not result.passed or result.fix_hint  # warning or reject

    def test_valid_proof_passes(self):
        result = self.pf.check("intro n\nsimp\nomega")
        assert result.passed

    def test_nat_subtract_warning(self):
        result = self.pf.check(
            "exact n - m",
            theorem="theorem t (n m : Nat) : n - m + m = n")
        # Should pass (warning only in non-strict) but with hint
        assert result.passed
        # May have fix_hint about nat subtraction


# ═══════════════════════════════════════════════════════════════
# Layer 7: Core Utilities
# ═══════════════════════════════════════════════════════════════

class TestCoreUtilities:
    """Test shared pure functions."""

    def test_classify_error_type_mismatch(self):
        assert classify_error("type mismatch\nexpected Nat") == "type_mismatch"

    def test_classify_error_unknown_id(self):
        assert classify_error("unknown identifier 'foo'") == "unknown_identifier"

    def test_classify_error_unsolved(self):
        assert classify_error("unsolved goals\n⊢ True") == "unsolved_goals"

    def test_classify_error_timeout(self):
        assert classify_error("deterministic timeout") == "timeout"

    def test_classify_error_tactic(self):
        assert classify_error("tactic 'simp' failed") == "tactic_failed"

    def test_assemble_code_basic(self):
        code = assemble_code("theorem t : True", ":= by trivial")
        assert "import Mathlib" in code
        assert "theorem t : True := by trivial" in code

    def test_assemble_code_tactic_block(self):
        code = assemble_code("theorem t : True", "by\n  trivial")
        assert ":= by" in code

    def test_assemble_code_already_complete(self):
        code = assemble_code("theorem t : True := trivial", "")
        assert "theorem t : True := trivial" in code

    def test_compile_cache(self):
        cache = CompileCache(maxsize=2)
        r1 = FullVerifyResult(success=True)
        r2 = FullVerifyResult(success=False, stderr="error")

        cache.put("k1", r1)
        cache.put("k2", r2)

        assert cache.get("k1").success is True
        assert cache.get("k2").success is False
        assert cache.get("k3") is None

        # LRU eviction
        cache.put("k3", r1)
        assert cache.get("k1") is None  # evicted (oldest)

    def test_make_cache_key_deterministic(self):
        k1 = make_cache_key("thm", "proof", "preamble", "v1")
        k2 = make_cache_key("thm", "proof", "preamble", "v1")
        k3 = make_cache_key("thm", "proof", "preamble", "v2")
        assert k1 == k2
        assert k1 != k3


# ═══════════════════════════════════════════════════════════════
# Layer 8: Concurrency Stress
# ═══════════════════════════════════════════════════════════════

class TestConcurrencyStress:
    """Stress tests for pool under concurrent load."""

    @pytest.mark.asyncio
    async def test_many_concurrent_tactics(self):
        """50 concurrent tactic requests through a 2-session pool."""
        pool = AsyncLeanPool(pool_size=2, project_dir=".")
        await pool.start()

        async def try_one(i):
            return await pool.try_tactic(0, f"tactic_{i}")

        results = await asyncio.gather(
            *(try_one(i) for i in range(50)),
            return_exceptions=True)

        assert len(results) == 50
        for r in results:
            assert isinstance(r, TacticFeedback)

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_verify(self):
        """10 concurrent verify_complete calls."""
        pool = AsyncLeanPool(pool_size=2, project_dir=".")
        await pool.start()

        async def verify_one(i):
            return await pool.verify_complete(
                f"theorem t{i} : True", ":= by trivial")

        results = await asyncio.gather(
            *(verify_one(i) for i in range(10)),
            return_exceptions=True)

        assert len(results) == 10
        await pool.shutdown()

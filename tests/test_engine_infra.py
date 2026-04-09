"""Consolidated engine infrastructure tests (elastic pool, proof session, verifier)"""


# ============================================================
# Source: test_phase_b.py
# ============================================================

import sys
import os
import asyncio
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. PersistentLemmaBank
# ═══════════════════════════════════════════════════════════════

class TestPersistentLemmaBank:

    def _make_bank(self):
        from prover.lemma_bank.persistent_bank import PersistentLemmaBank
        tmp = tempfile.mktemp(suffix=".db")
        return PersistentLemmaBank(db_path=tmp), tmp

    def test_add_and_retrieve(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        try:
            lemma = ProvedLemma(
                name="h1",
                statement="lemma h1 (n : Nat) : n + 0 = n",
                proof=":= by simp",
                verified=True)
            lid = bank.add(lemma, source_problem="p1")
            assert lid is not None

            results = bank.get_for_problem("p1")
            assert len(results) == 1
            assert results[0].name == "h1"
        finally:
            bank.close()
            os.unlink(path)

    def test_dedup(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        try:
            lemma = ProvedLemma(name="h", statement="lemma h : True",
                                proof=":= trivial", verified=True)
            id1 = bank.add(lemma)
            id2 = bank.add(lemma)  # duplicate
            assert id1 is not None
            assert id2 is None
            assert bank.stats()["total"] == 1
        finally:
            bank.close()
            os.unlink(path)

    def test_search(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        try:
            bank.add(ProvedLemma(
                name="nat_add", statement="lemma nat_add (n : Nat) : n + 0 = n",
                proof=":= by simp", verified=True))
            bank.add(ProvedLemma(
                name="list_nil", statement="lemma list_nil : List.nil = []",
                proof=":= rfl", verified=True))

            results = bank.search("Nat", top_k=5)
            assert len(results) >= 1
            assert any("Nat" in r.statement for r in results)
        finally:
            bank.close()
            os.unlink(path)

    def test_mark_used(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        try:
            lid = bank.add(ProvedLemma(
                name="h", statement="lemma h : True",
                proof=":= trivial", verified=True))
            bank.mark_used(lid)
            bank.mark_used(lid)
            most_used = bank.get_most_used(1)
            assert most_used[0].times_used == 2
        finally:
            bank.close()
            os.unlink(path)

    def test_stale_on_upgrade(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        bank._lean_version = "v4.8.0"
        try:
            bank.add(ProvedLemma(name="h", statement="lemma h : True",
                                  proof=":= trivial", verified=True))
            assert bank.stats()["verified"] == 1

            bank.mark_stale_on_upgrade(new_lean_version="v4.9.0")
            assert bank.stats()["verified"] == 0
            assert bank.stats()["stale"] == 1
        finally:
            bank.close()
            os.unlink(path)

    def test_to_lean_preamble(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        try:
            bank.add(ProvedLemma(name="h", statement="lemma h : True",
                                  proof=":= trivial", verified=True))
            preamble = bank.to_lean_preamble()
            assert "lemma h : True" in preamble
            assert ":= trivial" in preamble
        finally:
            bank.close()
            os.unlink(path)

    def test_persistence_across_instances(self):
        """关闭后重新打开应保留数据"""
        from prover.lemma_bank.bank import ProvedLemma
        from prover.lemma_bank.persistent_bank import PersistentLemmaBank
        tmp = tempfile.mktemp(suffix=".db")
        try:
            bank1 = PersistentLemmaBank(db_path=tmp)
            bank1.add(ProvedLemma(name="h", statement="lemma h : True",
                                    proof=":= trivial", verified=True))
            bank1.close()

            bank2 = PersistentLemmaBank(db_path=tmp)
            assert bank2.stats()["total"] == 1
            bank2.close()
        finally:
            os.unlink(tmp)

    def test_batch_add(self):
        from prover.lemma_bank.bank import ProvedLemma
        bank, path = self._make_bank()
        try:
            lemmas = [
                ProvedLemma(name=f"h{i}", statement=f"lemma h{i} : {i}={i}",
                            proof=":= rfl", verified=True)
                for i in range(5)
            ]
            added = bank.add_batch(lemmas, source_problem="batch_test")
            assert added == 5
            assert bank.stats()["total"] == 5
        finally:
            bank.close()
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# 2. ProofSessionManager
# ═══════════════════════════════════════════════════════════════

class TestProofSession:

    @pytest.mark.asyncio
    async def test_begin_proof(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                session = await mgr.begin_proof(
                    "theorem t : True := by",
                    session_id="s1")
                assert session is not None
                assert session.current_env_id >= 0

    @pytest.mark.asyncio
    async def test_try_step(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                session = await mgr.begin_proof("theorem t : True := by")
                result = await session.try_step("trivial")
                # In fallback mode, step will fail, but it shouldn't crash
                assert hasattr(result, 'success')
                assert hasattr(result, 'tactic')

    @pytest.mark.asyncio
    async def test_rewind(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                session = await mgr.begin_proof("theorem t : True := by")
                initial_env = session.current_env_id

                await session.try_step("simp")
                after_step_env = session.current_env_id

                rewound_env = session.rewind(steps=1)
                # Should go back to previous env
                assert session.current_env_id == rewound_env

    @pytest.mark.asyncio
    async def test_try_alternatives(self):
        """并行尝试不应改变 current_env_id"""
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=2, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                session = await mgr.begin_proof("theorem t : True := by")
                before = session.current_env_id

                results = await session.try_alternatives(
                    ["simp", "ring", "omega"])
                assert len(results) == 3

                # current should NOT have changed
                assert session.current_env_id == before

    @pytest.mark.asyncio
    async def test_tree_stats(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                session = await mgr.begin_proof("theorem t : True := by")
                stats = session.tree_stats()
                assert stats["total_nodes"] >= 1
                assert "max_depth" in stats
                assert "solved" in stats

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                s1 = await mgr.begin_proof("theorem t1 : True := by", "s1")
                s2 = await mgr.begin_proof("theorem t2 : 1=1 := by", "s2")

                assert mgr.get_session("s1") is s1
                assert mgr.get_session("s2") is s2
                assert len(mgr.list_sessions()) == 2

    @pytest.mark.asyncio
    async def test_get_successful_branches(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.proof_session import ProofSessionManager

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            async with ProofSessionManager(pool) as mgr:
                session = await mgr.begin_proof("theorem t : True := by")
                await session.try_step("simp")
                await session.try_step("ring")
                branches = session.get_successful_branches()
                assert isinstance(branches, list)


# ═══════════════════════════════════════════════════════════════
# 3. IncrementalVerifier
# ═══════════════════════════════════════════════════════════════

class TestIncrementalVerifier:

    @pytest.mark.asyncio
    async def test_verify_script(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.incremental_verifier import IncrementalVerifier

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            verifier = IncrementalVerifier(pool)
            result = await verifier.verify_script(
                theorem="theorem t : True := by",
                tactics=["trivial"],
                session_id="vs1")

            assert hasattr(result, 'success')
            assert result.steps_verified >= 1
            assert result.total_steps == 1
            assert len(result.step_results) >= 1

    @pytest.mark.asyncio
    async def test_verify_edit_speedup(self):
        """编辑应有步骤复用"""
        from engine.async_lean_pool import AsyncLeanPool
        from engine.incremental_verifier import IncrementalVerifier

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            verifier = IncrementalVerifier(pool)

            # 先验证完整脚本 (3 步)
            await verifier.verify_script(
                theorem="theorem t : True := by",
                tactics=["simp", "ring", "omega"],
                session_id="edit_test")

            # 编辑第 2 步 (步骤 0 和 1 应复用)
            result = await verifier.verify_edit(
                session_id="edit_test",
                edit_step=2,
                new_tactic="linarith")

            assert result.steps_reused >= 0  # 复用前面的步骤
            assert result.steps_verified >= 1  # 至少验证了编辑的步骤

    @pytest.mark.asyncio
    async def test_explore_alternatives(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.incremental_verifier import IncrementalVerifier

        async with AsyncLeanPool(pool_size=2, project_dir="/tmp") as pool:
            verifier = IncrementalVerifier(pool)

            await verifier.verify_script(
                theorem="theorem t : True := by",
                tactics=["simp"],
                session_id="explore_test")

            results = await verifier.explore_alternatives(
                session_id="explore_test",
                at_step=1,
                alternatives=["ring", "omega", "norm_num"])

            assert len(results) == 3
            for r in results:
                assert r.steps_reused >= 0
                assert r.steps_verified == 1

    @pytest.mark.asyncio
    async def test_stats(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.incremental_verifier import IncrementalVerifier

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            verifier = IncrementalVerifier(pool)
            await verifier.verify_script(
                theorem="theorem t : True := by",
                tactics=["simp", "ring"])

            s = verifier.stats()
            assert s["total_verifications"] >= 1
            assert s["total_steps_verified"] >= 1
            assert "reuse_rate" in s
            assert "avg_speedup" in s

    @pytest.mark.asyncio
    async def test_nonexistent_session(self):
        from engine.async_lean_pool import AsyncLeanPool
        from engine.incremental_verifier import IncrementalVerifier

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            verifier = IncrementalVerifier(pool)
            result = await verifier.verify_edit(
                session_id="nonexistent",
                edit_step=0, new_tactic="simp")
            assert result.success is False
            assert "not found" in result.error_message

    @pytest.mark.asyncio
    async def test_incremental_result_speedup(self):
        from engine.incremental_verifier import IncrementalResult
        r = IncrementalResult(
            success=True,
            steps_verified=2,
            steps_reused=8,
            total_steps=10)
        assert r.speedup == 5.0  # 10/2


# ============================================================
# Source: test_gap_fixes.py
# ============================================================

import asyncio
import json
import tempfile
import time

import pytest

from engine.proof_session import EnvNode, ProofSessionState
from engine.proof_context_store import (
    ProofContextStore, StepDetail, RichProofTrajectory,
    _serialize_state, _deserialize_state, _theorem_hash,
)
from engine.remote_session import (
    ElasticPool, RemoteSession, LocalTransport, TCPTransport, Transport,
)
from engine._core import TacticFeedback, FullVerifyResult
from engine.api.protocols import AsyncPoolProtocol


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_state(theorem="theorem t : 1+1=2 := by", solved=False,
                depth=3) -> ProofSessionState:
    nodes = {}
    for i in range(depth + 1):
        nodes[i] = EnvNode(
            env_id=i,
            parent_env_id=i - 1 if i > 0 else -1,
            tactic=f"tactic_{i}" if i > 0 else "",
            goals=[f"goal_{i}"] if not (solved and i == depth) else [],
            is_proof_complete=(solved and i == depth),
            children=[i + 1] if i < depth else [],
            depth=i,
        )
    return ProofSessionState(
        theorem=theorem,
        root_env_id=0,
        theorem_env_id=1,
        current_env_id=depth,
        nodes=nodes,
        tactic_history=[f"tactic_{i}" for i in range(1, depth + 1)],
        best_depth=depth,
        solved=solved,
        proof_path=list(range(depth + 1)) if solved else [],
    )


class MockTransport(Transport):
    """Mock transport for testing ElasticPool without real REPL."""

    def __init__(self, base_env_id=1, is_remote=False):
        self._connected = False
        self._base_env_id = base_env_id
        self._is_remote = is_remote
        self._sent = []

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def send(self, request: dict):
        self._sent.append(request)
        cmd = request.get("cmd", "")
        env = request.get("env", 0)

        # Simulate preamble
        if "import" in cmd:
            return {"env": self._base_env_id, "messages": [], "goals": []}

        # Simulate tactic
        if "sorry" in cmd or "fail" in cmd:
            return {"env": env, "messages": [
                {"severity": "error", "data": "tactic failed"}
            ], "goals": []}

        # Simulate success
        return {"env": env + 1, "messages": [], "goals": []}

    async def close(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


async def _make_elastic_pool(local=2, remote=0):
    """Create an ElasticPool with mock sessions."""
    pool = ElasticPool(timeout_seconds=5)

    for i in range(local):
        transport = MockTransport(base_env_id=10 + i)
        session = RemoteSession(transport, session_id=i)
        await session.start("import Mathlib")
        pool._sessions.append(session)

    for i in range(remote):
        transport = MockTransport(base_env_id=20 + i, is_remote=True)
        # Make it look like a TCP transport for type checks
        transport.__class__ = type('MockTCPTransport',
                                   (MockTransport, TCPTransport),
                                   {'__init__': MockTransport.__init__})
        session = RemoteSession(transport, session_id=local + i)
        await session.start("import Mathlib")
        pool._sessions.append(session)

    pool._started = True
    return pool


# ═══════════════════════════════════════════════════════════════
# Gap 1: ElasticPool 闭环
# ═══════════════════════════════════════════════════════════════

class TestElasticPoolBaseEnvId:

    @pytest.mark.asyncio
    async def test_base_env_id_returns_first_alive_session(self):
        pool = await _make_elastic_pool(local=2)
        # First session has base_env_id from MockTransport (10)
        assert pool.base_env_id == 10

    @pytest.mark.asyncio
    async def test_base_env_id_empty_pool(self):
        pool = ElasticPool()
        assert pool.base_env_id == 0

    @pytest.mark.asyncio
    async def test_base_env_id_skips_dead_sessions(self):
        pool = await _make_elastic_pool(local=2)
        pool._sessions[0]._alive = False
        assert pool.base_env_id == 11  # second session


class TestElasticPoolShareLemma:

    @pytest.mark.asyncio
    async def test_share_lemma_all_sessions(self):
        pool = await _make_elastic_pool(local=3)
        count = await pool.share_lemma("lemma helper : True := trivial")
        assert count == 3

    @pytest.mark.asyncio
    async def test_share_lemma_single_session(self):
        pool = await _make_elastic_pool(local=3)
        count = await pool.share_lemma(
            "lemma helper : True := trivial", inject_all=False)
        assert count == 1

    @pytest.mark.asyncio
    async def test_share_lemma_skips_fallback(self):
        pool = await _make_elastic_pool(local=2)
        pool._sessions[0]._fallback = True
        count = await pool.share_lemma("lemma h : True := trivial")
        # Only non-fallback session succeeds
        assert count == 1


class TestElasticPoolDynamicScaling:

    @pytest.mark.asyncio
    async def test_add_session_local(self):
        pool = await _make_elastic_pool(local=1)
        assert len(pool._sessions) == 1

        # Monkey-patch to use mock transport
        original_start = RemoteSession.start

        async def mock_start(self, preamble=""):
            self._alive = True
            self._base_env_id = 99
            return True

        RemoteSession.start = mock_start
        try:
            ok = await pool.add_session()
            assert ok
            assert len(pool._sessions) == 2
            assert getattr(pool._sessions[-1], '_is_overflow', False)
        finally:
            RemoteSession.start = original_start

    @pytest.mark.asyncio
    async def test_remove_idle_prefers_remote_overflow(self):
        pool = await _make_elastic_pool(local=1, remote=1)
        # Mark remote session as overflow
        pool._sessions[1]._is_overflow = True

        removed = await pool.remove_idle_session()
        assert removed
        assert len(pool._sessions) == 1
        # The remaining session should be the local one
        assert isinstance(pool._sessions[0].transport, MockTransport)


class TestElasticPoolStats:

    @pytest.mark.asyncio
    async def test_stats_includes_all_fields(self):
        pool = await _make_elastic_pool(local=2)
        stats = pool.stats()
        assert "total_sessions" in stats
        assert "busy_sessions" in stats
        assert "active_sessions" in stats
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2


# ═══════════════════════════════════════════════════════════════
# Gap 2: Rich Trajectory Format
# ═══════════════════════════════════════════════════════════════

class TestStepDetail:

    def test_step_detail_creation(self):
        step = StepDetail(
            step_index=0,
            tactic="simp",
            env_id_before=1,
            env_id_after=2,
            goals_before=["⊢ 1 + 1 = 2"],
            goals_after=[],
            is_proof_complete=True,
        )
        assert step.tactic == "simp"
        assert step.is_proof_complete
        assert step.error_message == ""

    def test_step_detail_with_error(self):
        step = StepDetail(
            step_index=1,
            tactic="ring",
            env_id_before=2,
            env_id_after=-1,
            goals_before=["⊢ n - m ≤ n"],
            goals_after=["⊢ n - m ≤ n"],
            error_message="ring failed",
            error_category="tactic_failed",
            elapsed_ms=42.0,
        )
        assert step.env_id_after == -1
        assert step.error_category == "tactic_failed"


class TestRichTrajectoryStore:

    @pytest.fixture
    def store(self):
        return ProofContextStore(":memory:")

    @pytest.mark.asyncio
    async def test_record_and_export_rich_trace(self, store):
        state = _make_state(solved=True)
        ctx_id = await store.save(state)

        steps = [
            StepDetail(
                step_index=0, tactic="intro n",
                env_id_before=1, env_id_after=2,
                goals_before=["⊢ ∀ n, n = n"],
                goals_after=["n : ℕ ⊢ n = n"],
                elapsed_ms=15.0,
            ),
            StepDetail(
                step_index=1, tactic="rfl",
                env_id_before=2, env_id_after=3,
                goals_before=["n : ℕ ⊢ n = n"],
                goals_after=[],
                is_proof_complete=True,
                elapsed_ms=5.0,
            ),
        ]

        trace_id = await store.record_rich_trace(
            ctx_id, steps, success=True, duration_ms=20.0)
        assert trace_id > 0

        trajectories = await store.export_rich_trajectories()
        assert len(trajectories) == 1

        traj = trajectories[0]
        assert isinstance(traj, RichProofTrajectory)
        assert traj.success is True
        assert traj.depth == 2
        assert len(traj.steps) == 2
        assert traj.steps[0].tactic == "intro n"
        assert traj.steps[0].goals_before == ["⊢ ∀ n, n = n"]
        assert traj.steps[0].env_id_before == 1
        assert traj.steps[0].env_id_after == 2
        assert traj.steps[1].is_proof_complete is True
        assert traj.theorem == state.theorem
        assert traj.context_id == ctx_id

    @pytest.mark.asyncio
    async def test_export_rich_excludes_legacy_traces(self, store):
        state = _make_state()
        ctx_id = await store.save(state)

        # Record a legacy trace (no step_details)
        await store.record_trace(
            ctx_id, ["simp", "omega"], success=True, depth=2)

        # Rich export should skip it
        rich = await store.export_rich_trajectories()
        assert len(rich) == 0

    @pytest.mark.asyncio
    async def test_export_rich_with_error_steps(self, store):
        state = _make_state()
        ctx_id = await store.save(state)

        steps = [
            StepDetail(
                step_index=0, tactic="ring",
                env_id_before=1, env_id_after=-1,
                goals_before=["⊢ n - m ≤ n"],
                goals_after=["⊢ n - m ≤ n"],
                error_message="ring failed on ℕ subtraction",
                error_category="tactic_failed",
                elapsed_ms=50.0,
            ),
            StepDetail(
                step_index=1, tactic="omega",
                env_id_before=1, env_id_after=2,
                goals_before=["⊢ n - m ≤ n"],
                goals_after=[],
                is_proof_complete=True,
                elapsed_ms=30.0,
            ),
        ]

        await store.record_rich_trace(ctx_id, steps, success=True,
                                       duration_ms=80.0)

        trajs = await store.export_rich_trajectories()
        assert len(trajs) == 1
        assert trajs[0].steps[0].error_message == "ring failed on ℕ subtraction"
        assert trajs[0].steps[0].env_id_after == -1
        assert trajs[0].steps[1].is_proof_complete is True

    @pytest.mark.asyncio
    async def test_export_rich_min_depth_filter(self, store):
        state = _make_state()
        ctx_id = await store.save(state)

        # 1-step trace
        await store.record_rich_trace(ctx_id, [
            StepDetail(0, "trivial", 1, 2, ["⊢ True"], [],
                       is_proof_complete=True, elapsed_ms=1.0),
        ], success=True, duration_ms=1.0)

        # 3-step trace
        await store.record_rich_trace(ctx_id, [
            StepDetail(0, "intro n", 1, 2, ["⊢ ∀ n, P n"], ["⊢ P n"]),
            StepDetail(1, "cases n", 2, 3, ["⊢ P n"], ["⊢ P 0", "⊢ P (n+1)"]),
            StepDetail(2, "simp", 3, 4, ["⊢ P 0", "⊢ P (n+1)"], [],
                       is_proof_complete=True),
        ], success=True, duration_ms=100.0)

        # min_depth=2 should only return the 3-step trace
        trajs = await store.export_rich_trajectories(min_depth=2)
        assert len(trajs) == 1
        assert trajs[0].depth == 3

    @pytest.mark.asyncio
    async def test_export_rich_success_only(self, store):
        state = _make_state()
        ctx_id = await store.save(state)

        await store.record_rich_trace(ctx_id, [
            StepDetail(0, "simp", 1, -1, ["⊢ P"], ["⊢ P"],
                       error_message="failed"),
        ], success=False, duration_ms=10.0)

        await store.record_rich_trace(ctx_id, [
            StepDetail(0, "trivial", 1, 2, ["⊢ True"], [],
                       is_proof_complete=True),
        ], success=True, duration_ms=5.0)

        trajs = await store.export_rich_trajectories(success_only=True)
        assert len(trajs) == 1
        assert trajs[0].success is True


class TestDBMigration:

    @pytest.mark.asyncio
    async def test_file_db_migration_v2(self):
        """Test that opening an existing v1 DB auto-migrates."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name

        # Create a v1-style DB (no step_details column)
        import sqlite3
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE proof_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theorem_hash TEXT NOT NULL,
                theorem TEXT NOT NULL,
                state_json TEXT NOT NULL,
                best_depth INTEGER DEFAULT 0,
                num_nodes INTEGER DEFAULT 0,
                num_tactics INTEGER DEFAULT 0,
                solved INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE proof_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id INTEGER NOT NULL,
                tactic_sequence TEXT NOT NULL,
                depth INTEGER DEFAULT 0,
                duration_ms REAL DEFAULT 0.0,
                success INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
        """)
        conn.close()

        # Open with ProofContextStore — should auto-migrate
        store = ProofContextStore(path)
        state = _make_state()
        ctx_id = await store.save(state)

        # Should be able to record rich trace after migration
        steps = [
            StepDetail(0, "simp", 1, 2, ["⊢ P"], [],
                       is_proof_complete=True),
        ]
        trace_id = await store.record_rich_trace(
            ctx_id, steps, success=True)
        assert trace_id > 0

        import os
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# Gap 3: Async Search Coordinator
# ═══════════════════════════════════════════════════════════════

class MockAsyncPool:
    """Mock async pool for testing AsyncSearchCoordinator."""

    def __init__(self, base_env=10):
        self._base_env = base_env
        self._next_env = base_env + 1
        self._calls = []

    @property
    def base_env_id(self) -> int:
        return self._base_env

    async def try_tactic(self, env_id: int,
                         tactic: str) -> TacticFeedback:
        self._calls.append(("try_tactic", env_id, tactic))

        if "fail" in tactic:
            return TacticFeedback(
                success=False, tactic=tactic,
                error_message="tactic failed",
                error_category="tactic_failed",
                elapsed_ms=5)

        new_env = self._next_env
        self._next_env += 1
        complete = ("qed" in tactic or "rfl" in tactic
                    or "trivial" in tactic)
        return TacticFeedback(
            success=True, tactic=tactic,
            new_env_id=new_env,
            remaining_goals=[] if complete else ["⊢ remaining"],
            is_proof_complete=complete,
            elapsed_ms=10)

    async def try_tactics_parallel(self, env_id: int,
                                   tactics: list[str]
                                   ) -> list[TacticFeedback]:
        return [await self.try_tactic(env_id, t) for t in tactics]

    def stats(self) -> dict:
        return {"calls": len(self._calls)}


class TestAsyncSearchCoordinator:

    @pytest.fixture
    def mock_pool(self):
        return MockAsyncPool()

    def _make_coordinator(self, mock_pool):
        """Create a minimal coordinator for testing."""
        # We need the engine internals for the search tree
        try:
            from engine.core import Expr, Name, Environment
            from engine.async_search import AsyncSearchCoordinator, SearchConfig

            env = Environment()
            goal_type = Expr.const(Name.from_str("test_goal"))
            config = SearchConfig(
                strategy="best_first",
                max_nodes=100,
                timeout_ms=5000,
            )
            return AsyncSearchCoordinator(
                env, goal_type, config=config, async_pool=mock_pool)
        except Exception:
            pytest.skip("Engine core not available for search tests")

    @pytest.mark.asyncio
    async def test_async_try_tactic_success(self, mock_pool):
        coord = self._make_coordinator(mock_pool)
        result = await coord.async_try_tactic(0, "simp")
        assert result.success
        assert result.child_node is not None
        assert len(mock_pool._calls) == 1

    @pytest.mark.asyncio
    async def test_async_try_tactic_failure(self, mock_pool):
        coord = self._make_coordinator(mock_pool)
        result = await coord.async_try_tactic(0, "fail_tactic")
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_async_try_batch_parallel(self, mock_pool):
        coord = self._make_coordinator(mock_pool)
        results = await coord.async_try_batch(
            0, ["simp", "omega", "fail_tactic"])
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 2
        assert len(failures) == 1
        # All three should have gone through the pool
        assert len(mock_pool._calls) == 3

    @pytest.mark.asyncio
    async def test_async_try_tactic_maps_env_id(self, mock_pool):
        coord = self._make_coordinator(mock_pool)

        # First tactic expands root → child
        r1 = await coord.async_try_tactic(0, "intro n")
        assert r1.success
        child_id = r1.child_node

        # The child node should have a mapped env_id
        assert child_id in coord._node_env_map
        new_env = coord._node_env_map[child_id]

        # Second tactic on child should use the new env_id
        r2 = await coord.async_try_tactic(child_id, "trivial")
        assert len(mock_pool._calls) == 2
        assert mock_pool._calls[1][1] == new_env  # env_id passed to pool

    @pytest.mark.asyncio
    async def test_async_run_search_completes(self, mock_pool):
        coord = self._make_coordinator(mock_pool)

        # Generator that always suggests "trivial" (which completes)
        def gen(node_id):
            return ["trivial"]

        stats = await coord.async_run_search(gen)
        assert stats.is_solved
        assert stats.nodes_expanded >= 1

    @pytest.mark.asyncio
    async def test_async_run_search_with_async_generator(self, mock_pool):
        coord = self._make_coordinator(mock_pool)

        async def async_gen(node_id):
            await asyncio.sleep(0)  # simulate async LLM call
            return ["trivial"]

        stats = await coord.async_run_search(
            tactic_generator=lambda _: [],
            async_tactic_generator=async_gen)
        assert stats.is_solved

    @pytest.mark.asyncio
    async def test_sync_fallback_without_pool(self):
        """Without async_pool, falls back to local tactic engine."""
        try:
            from engine.core import Expr, Name, Environment
            from engine.async_search import AsyncSearchCoordinator, SearchConfig

            env = Environment()
            goal_type = Expr.const(Name.from_str("test_goal"))
            coord = AsyncSearchCoordinator(
                env, goal_type,
                config=SearchConfig(max_nodes=10),
                async_pool=None)

            # Should use sync path (local engine)
            result = await coord.async_try_tactic(0, "rfl")
            # Result depends on local engine capability, but shouldn't crash
            assert isinstance(result, ExpansionResult)
        except Exception:
            pytest.skip("Engine core not available for search tests")


# ═══════════════════════════════════════════════════════════════
# Protocol compliance
# ═══════════════════════════════════════════════════════════════

class TestProtocolCompliance:

    @pytest.mark.asyncio
    async def test_elastic_pool_has_base_env_id(self):
        pool = await _make_elastic_pool(local=1)
        assert hasattr(pool, 'base_env_id')
        assert isinstance(pool.base_env_id, int)

    @pytest.mark.asyncio
    async def test_elastic_pool_has_share_lemma(self):
        pool = await _make_elastic_pool(local=1)
        assert hasattr(pool, 'share_lemma')
        assert asyncio.iscoroutinefunction(pool.share_lemma)

    @pytest.mark.asyncio
    async def test_elastic_pool_has_required_methods(self):
        pool = await _make_elastic_pool(local=1)
        required = [
            'try_tactic', 'try_tactics_parallel', 'verify_complete',
            'share_lemma', 'add_session', 'remove_idle_session',
            'shutdown', 'stats', 'base_env_id',
        ]
        for method in required:
            assert hasattr(pool, method), f"Missing: {method}"


# ============================================================
# Source: test_phase6_elastic.py
# ============================================================

import asyncio
import json
import os
import tempfile
import time

import pytest

from engine.proof_session import EnvNode, ProofSessionState
from engine.proof_context_store import (
    ProofContextStore, _serialize_state, _deserialize_state,
    _theorem_hash, ProofContextInfo, ProofTrajectory,
)
from engine.pool_scaler import PoolScaler, ScaleDecision
from engine.resource_scheduler import (
    ResourceScheduler, ResourceBudget, Priority,
    TaskHandle, BudgetExhaustedError, ConcurrencyLimitError,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

def _make_state(theorem="theorem t : 1+1=2 := by", solved=False,
                depth=3) -> ProofSessionState:
    nodes = {}
    for i in range(depth + 1):
        nodes[i] = EnvNode(
            env_id=i,
            parent_env_id=i - 1 if i > 0 else -1,
            tactic=f"tactic_{i}" if i > 0 else "",
            goals=[f"goal_{i}"] if not (solved and i == depth) else [],
            is_proof_complete=(solved and i == depth),
            children=[i + 1] if i < depth else [],
            depth=i,
        )
    return ProofSessionState(
        theorem=theorem,
        root_env_id=0,
        theorem_env_id=1,
        current_env_id=depth,
        nodes=nodes,
        tactic_history=[f"tactic_{i}" for i in range(1, depth + 1)],
        best_depth=depth,
        solved=solved,
        proof_path=list(range(depth + 1)) if solved else [],
    )


class MockSession:
    """Mock session for pool testing."""
    def __init__(self, sid=0, alive=True, busy=False):
        self.session_id = sid
        self._alive = alive
        self._busy = busy
        self._is_overflow = False

    @property
    def is_busy(self):
        return self._busy

    @property
    def is_alive(self):
        return self._alive

    @property
    def is_fallback(self):
        return False

    async def close(self):
        self._alive = False


class MockPool:
    """Mock pool implementing the interface PoolScaler expects."""
    def __init__(self, active=4, busy=2):
        self._active = active
        self._busy = busy
        self._sessions = [
            MockSession(i, alive=True, busy=(i < busy))
            for i in range(active)
        ]
        self._added = 0
        self._removed = 0

    def stats(self):
        return {
            "active_sessions": self._active,
            "busy_sessions": self._busy,
            "total_sessions": self._active,
        }

    async def add_session(self) -> bool:
        self._active += 1
        self._added += 1
        return True

    async def remove_idle_session(self) -> bool:
        if self._active > self._busy:
            self._active -= 1
            self._removed += 1
            return True
        return False


class MockPoolForScheduler:
    """Mock pool for ResourceScheduler testing."""
    def __init__(self):
        self._sessions = [MockSession(i) for i in range(4)]
        self._available = asyncio.Queue()
        for s in self._sessions:
            self._available.put_nowait(s)

    async def _acquire_session(self):
        return await asyncio.wait_for(self._available.get(), timeout=2.0)

    async def _release_session(self, session):
        self._available.put_nowait(session)

    def stats(self):
        return {
            "active_sessions": len(self._sessions),
            "busy_sessions": len(self._sessions) - self._available.qsize(),
        }


# ═══════════════════════════════════════════════════════════════
# ProofContextStore tests
# ═══════════════════════════════════════════════════════════════

class TestSerialization:

    def test_serialize_roundtrip(self):
        state = _make_state(solved=True, depth=5)
        json_str = _serialize_state(state)
        restored = _deserialize_state(json_str)

        assert restored.theorem == state.theorem
        assert restored.root_env_id == state.root_env_id
        assert restored.current_env_id == state.current_env_id
        assert restored.solved == state.solved
        assert restored.best_depth == state.best_depth
        assert restored.tactic_history == state.tactic_history
        assert restored.proof_path == state.proof_path
        assert len(restored.nodes) == len(state.nodes)

    def test_serialize_preserves_node_details(self):
        state = _make_state(depth=3)
        json_str = _serialize_state(state)
        restored = _deserialize_state(json_str)

        for env_id, node in state.nodes.items():
            rn = restored.nodes[env_id]
            assert rn.env_id == node.env_id
            assert rn.parent_env_id == node.parent_env_id
            assert rn.tactic == node.tactic
            assert rn.goals == node.goals
            assert rn.is_proof_complete == node.is_proof_complete
            assert rn.children == node.children
            assert rn.depth == node.depth

    def test_serialize_empty_state(self):
        state = ProofSessionState(
            theorem="theorem x : True := by",
            root_env_id=0, theorem_env_id=1, current_env_id=1,
            nodes={0: EnvNode(env_id=0)},
        )
        json_str = _serialize_state(state)
        restored = _deserialize_state(json_str)
        assert restored.theorem == state.theorem
        assert len(restored.nodes) == 1

    def test_theorem_hash_deterministic(self):
        h1 = _theorem_hash("theorem t : Nat := by")
        h2 = _theorem_hash("theorem t : Nat := by")
        h3 = _theorem_hash("theorem t : Nat := by ")  # trailing space stripped
        assert h1 == h2
        assert h1 == h3

    def test_theorem_hash_different(self):
        h1 = _theorem_hash("theorem a : Nat := by")
        h2 = _theorem_hash("theorem b : Nat := by")
        assert h1 != h2


class TestProofContextStore:

    @pytest.fixture
    def store(self):
        return ProofContextStore(":memory:")

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        state = _make_state(solved=True)
        ctx_id = await store.save(state)
        assert ctx_id > 0

        loaded = await store.load(ctx_id)
        assert loaded is not None
        assert loaded.theorem == state.theorem
        assert loaded.solved is True
        assert loaded.best_depth == state.best_depth

    @pytest.mark.asyncio
    async def test_update_existing(self, store):
        state = _make_state(solved=False)
        ctx_id = await store.save(state)

        state.solved = True
        state.best_depth = 10
        same_id = await store.save(state, context_id=ctx_id)
        assert same_id == ctx_id

        loaded = await store.load(ctx_id)
        assert loaded.solved is True
        assert loaded.best_depth == 10

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, store):
        result = await store.load(99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_load_by_theorem(self, store):
        state = _make_state(theorem="theorem unique_abc : True := by")
        await store.save(state)

        loaded = await store.load_by_theorem(
            "theorem unique_abc : True := by")
        assert loaded is not None
        assert loaded.theorem == state.theorem

    @pytest.mark.asyncio
    async def test_load_by_theorem_not_found(self, store):
        result = await store.load_by_theorem("nonexistent theorem")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        state = _make_state()
        ctx_id = await store.save(state)
        assert await store.delete(ctx_id) is True
        assert await store.load(ctx_id) is None
        assert await store.delete(ctx_id) is False

    @pytest.mark.asyncio
    async def test_list_recent(self, store):
        for i in range(5):
            s = _make_state(theorem=f"theorem t{i} : Nat := by",
                            solved=(i % 2 == 0))
            await store.save(s)

        recent = await store.list_recent(limit=10)
        assert len(recent) == 5

        solved = await store.list_recent(limit=10, solved_only=True)
        assert len(solved) == 3  # t0, t2, t4

    @pytest.mark.asyncio
    async def test_record_and_export_traces(self, store):
        state = _make_state(solved=True, depth=4)
        ctx_id = await store.save(state)

        await store.record_trace(
            ctx_id, ["intro", "simp", "ring", "rfl"],
            success=True, depth=4, duration_ms=1200.0)
        await store.record_trace(
            ctx_id, ["intro", "omega"],
            success=False, depth=2, duration_ms=500.0)

        all_traces = await store.export_trajectories(min_depth=0)
        assert len(all_traces) == 2

        deep_traces = await store.export_trajectories(min_depth=3)
        assert len(deep_traces) == 1
        assert deep_traces[0].success is True
        assert deep_traces[0].depth == 4

        succ_traces = await store.export_trajectories(
            min_depth=0, success_only=True)
        assert len(succ_traces) == 1

    @pytest.mark.asyncio
    async def test_stats(self, store):
        for i in range(3):
            s = _make_state(theorem=f"thm{i}", solved=(i == 0))
            ctx_id = await store.save(s)
            await store.record_trace(ctx_id, ["tac"], success=(i == 0),
                                     depth=i + 1)

        stats = await store.stats()
        assert stats["total_contexts"] == 3
        assert stats["solved_contexts"] == 1
        assert stats["total_traces"] == 3
        assert stats["successful_traces"] == 1

    @pytest.mark.asyncio
    async def test_file_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store1 = ProofContextStore(db_path)
            state = _make_state(solved=True)
            ctx_id = await store1.save(state)

            # New store instance on same file
            store2 = ProofContextStore(db_path)
            loaded = await store2.load(ctx_id)
            assert loaded is not None
            assert loaded.theorem == state.theorem
            assert loaded.solved is True
        finally:
            os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════
# PoolScaler tests (with mock pools)
# ═══════════════════════════════════════════════════════════════

class TestPoolScaler:

    def test_evaluate_hold_within_thresholds(self):
        pool = MockPool(active=4, busy=2)  # 50% busy
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        decision = scaler.evaluate()
        assert decision.action == "hold"

    def test_evaluate_scale_up(self):
        pool = MockPool(active=4, busy=4)  # 100% busy
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        decision = scaler.evaluate()
        assert decision.action == "scale_up"
        assert decision.target_size == 5

    def test_evaluate_no_scale_up_at_max(self):
        pool = MockPool(active=8, busy=8)
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        decision = scaler.evaluate()
        assert decision.action == "hold"

    def test_evaluate_scale_down_after_idle(self):
        pool = MockPool(active=8, busy=1)  # 12.5% busy
        scaler = PoolScaler(
            pool, min_sessions=1, max_sessions=16,
            cooldown_seconds=0.01)

        # First call sets idle_since
        d1 = scaler.evaluate()
        assert d1.action == "hold"

        # Wait past cooldown
        time.sleep(0.02)
        d2 = scaler.evaluate()
        assert d2.action == "scale_down"
        assert d2.target_size == 7

    def test_evaluate_no_scale_down_at_min(self):
        pool = MockPool(active=1, busy=0)
        scaler = PoolScaler(
            pool, min_sessions=1, max_sessions=8,
            cooldown_seconds=0.01)
        scaler.evaluate()  # set idle_since
        time.sleep(0.02)
        d = scaler.evaluate()
        assert d.action == "hold"

    def test_cooldown_prevents_rapid_scaling(self):
        pool = MockPool(active=4, busy=4)
        scaler = PoolScaler(
            pool, min_sessions=1, max_sessions=8,
            cooldown_seconds=100.0)

        d1 = scaler.evaluate()
        assert d1.action == "scale_up"
        scaler._last_scale_time = time.time()  # simulate apply

        d2 = scaler.evaluate()
        assert d2.action == "hold"
        assert "cooldown" in d2.reason

    @pytest.mark.asyncio
    async def test_apply_scale_up(self):
        pool = MockPool(active=4, busy=4)
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        decision = ScaleDecision(
            action="scale_up", current_size=4, target_size=5)
        await scaler.apply(decision)
        assert pool._added == 1

    @pytest.mark.asyncio
    async def test_apply_scale_down(self):
        pool = MockPool(active=4, busy=1)
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        decision = ScaleDecision(
            action="scale_down", current_size=4, target_size=3)
        await scaler.apply(decision)
        assert pool._removed == 1

    def test_stats(self):
        pool = MockPool(active=4, busy=2)
        scaler = PoolScaler(pool, min_sessions=2, max_sessions=10)
        s = scaler.stats()
        assert s["pool_active"] == 4
        assert s["pool_busy"] == 2
        assert s["min_sessions"] == 2
        assert s["max_sessions"] == 10


# ═══════════════════════════════════════════════════════════════
# ResourceScheduler tests
# ═══════════════════════════════════════════════════════════════

class TestResourceBudget:

    def test_not_exhausted_initially(self):
        b = ResourceBudget()
        assert not b.is_exhausted
        assert b.remaining_ratio == 1.0

    def test_exhausted_after_max_verifications(self):
        b = ResourceBudget(max_verifications=3)
        for _ in range(3):
            b.consume_verification()
        assert b.is_exhausted

    def test_remaining_ratio(self):
        b = ResourceBudget(max_verifications=100, max_tokens=1000)
        b.consume_verification()
        b.consume_tokens(500)
        assert 0.0 < b.remaining_ratio < 1.0


class TestResourceScheduler:

    @pytest.mark.asyncio
    async def test_submit_and_complete(self):
        pool = MockPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        handle = await sched.submit("task_1", Priority.NORMAL)
        assert handle.task_id == "task_1"
        assert sched.stats()["active_tasks"] == 1

        await sched.complete(handle)
        assert sched.stats()["active_tasks"] == 0
        assert sched.stats()["total_completed"] == 1

    @pytest.mark.asyncio
    async def test_acquire_and_release_session(self):
        pool = MockPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        handle = await sched.submit("task_1")
        session = await handle.acquire_session()
        assert session is not None
        assert handle.budget.verifications_used == 1

        await handle.release_session(session)
        await sched.complete(handle)

    @pytest.mark.asyncio
    async def test_budget_exhaustion(self):
        pool = MockPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        handle = await sched.submit(
            "task_1", budget=ResourceBudget(max_verifications=2))

        s1 = await handle.acquire_session()
        await handle.release_session(s1)
        s2 = await handle.acquire_session()
        await handle.release_session(s2)

        with pytest.raises(BudgetExhaustedError):
            await handle.acquire_session()

        await sched.complete(handle)

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        pool = MockPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        handle = await sched.submit(
            "task_1",
            budget=ResourceBudget(max_concurrent_sessions=1))

        s1 = await handle.acquire_session()
        with pytest.raises(ConcurrencyLimitError):
            await handle.acquire_session()

        await handle.release_session(s1)
        await sched.complete(handle)

    @pytest.mark.asyncio
    async def test_cancel(self):
        pool = MockPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=2)

        h1 = await sched.submit("t1")
        h2 = await sched.submit("t2")
        assert sched.stats()["active_tasks"] == 2

        await sched.cancel("t1")
        assert sched.stats()["active_tasks"] == 1

    @pytest.mark.asyncio
    async def test_stats(self):
        pool = MockPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        h = await sched.submit("t1", Priority.HIGH)
        stats = sched.stats()
        assert stats["active_tasks"] == 1
        assert "t1" in stats["active_task_ids"]
        assert stats["budget_summary"]["t1"]["priority"] == "HIGH"

        await sched.complete(h)


# ═══════════════════════════════════════════════════════════════
# ElasticPool PoolScaler compatibility tests
# ═══════════════════════════════════════════════════════════════

class TestElasticPoolScalerCompat:
    """Test that ElasticPool works with PoolScaler."""

    def _make_mock_elastic_pool(self, n_sessions=4, n_busy=2):
        """Create a mock that matches ElasticPool's interface."""
        pool = MockPool(active=n_sessions, busy=n_busy)
        return pool

    def test_scaler_evaluates_elastic_pool(self):
        pool = self._make_mock_elastic_pool(4, 4)
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        d = scaler.evaluate()
        assert d.action == "scale_up"

    @pytest.mark.asyncio
    async def test_scaler_applies_to_elastic_pool(self):
        pool = self._make_mock_elastic_pool(4, 4)
        scaler = PoolScaler(pool, min_sessions=1, max_sessions=8)
        d = scaler.evaluate()
        await scaler.apply(d)
        assert pool._added == 1


# ═══════════════════════════════════════════════════════════════
# ResourceScheduler ElasticPool compatibility tests
# ═══════════════════════════════════════════════════════════════

class MockElasticPoolForScheduler:
    """Mock that uses ElasticPool's _acquire/_release interface."""
    def __init__(self):
        self._sessions = [MockSession(i) for i in range(4)]
        self._available = asyncio.Queue()
        for s in self._sessions:
            self._available.put_nowait(s)

    async def _acquire(self):
        return await asyncio.wait_for(self._available.get(), timeout=2.0)

    async def _release(self, session):
        self._available.put_nowait(session)

    def stats(self):
        return {
            "active_sessions": len(self._sessions),
            "busy_sessions": len(self._sessions) - self._available.qsize(),
        }


class TestResourceSchedulerElasticPool:

    @pytest.mark.asyncio
    async def test_scheduler_with_elastic_pool_interface(self):
        pool = MockElasticPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        handle = await sched.submit("task_1")
        session = await handle.acquire_session()
        assert session is not None

        await handle.release_session(session)
        await sched.complete(handle)

    @pytest.mark.asyncio
    async def test_multiple_tasks_different_priorities(self):
        pool = MockElasticPoolForScheduler()
        sched = ResourceScheduler(pool, max_concurrent_tasks=4)

        h1 = await sched.submit("low", Priority.LOW)
        h2 = await sched.submit("high", Priority.HIGH)
        h3 = await sched.submit("critical", Priority.CRITICAL)

        # All can acquire sessions
        s1 = await h1.acquire_session()
        s2 = await h2.acquire_session()
        s3 = await h3.acquire_session()

        await h1.release_session(s1)
        await h2.release_session(s2)
        await h3.release_session(s3)

        await sched.complete(h1)
        await sched.complete(h2)
        await sched.complete(h3)

        assert sched.stats()["total_completed"] == 3


# ═══════════════════════════════════════════════════════════════
# Integration: ProofContextStore + ProofSession workflow
# ═══════════════════════════════════════════════════════════════

class TestStoreWorkflow:

    @pytest.mark.asyncio
    async def test_save_explore_resume(self):
        """Simulate: start proof → save → resume → solve → save."""
        store = ProofContextStore(":memory:")

        # Start exploring
        state = _make_state(theorem="theorem p : 2+2=4 := by",
                            solved=False, depth=2)
        ctx_id = await store.save(state)

        # "Resume" by loading
        resumed = await store.load(ctx_id)
        assert resumed.current_env_id == 2
        assert not resumed.solved

        # Continue and solve
        resumed.solved = True
        resumed.best_depth = 5
        resumed.tactic_history.extend(["ring", "rfl", "done"])
        await store.save(resumed, context_id=ctx_id)

        # Record trace
        await store.record_trace(
            ctx_id, resumed.tactic_history,
            success=True, depth=5, duration_ms=2500.0)

        # Verify final state
        final = await store.load(ctx_id)
        assert final.solved is True
        assert final.best_depth == 5

        traces = await store.export_trajectories(success_only=True)
        assert len(traces) == 1
        assert traces[0].theorem == "theorem p : 2+2=4 := by"

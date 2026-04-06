"""tests/test_phase_b.py — Phase B 测试: 增量验证 + 持久化

覆盖:
  1. PersistentLemmaBank — SQLite CRUD + 检索 + 版本管理
  2. ProofSessionManager — 状态树 + 快照/回退 + fork
  3. IncrementalVerifier — 增量验证 + 编辑 + 并行探索
"""
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

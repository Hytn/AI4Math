"""tests/test_async_engine.py — 异步引擎层测试

覆盖:
  1. AsyncLeanSession / AsyncLeanPool — 并行 tactic, 会话管理
  2. AsyncVerificationScheduler — L0/L1 异步管线
  3. AsyncLLMProvider — 异步 LLM 调用
  4. AsyncSubAgent / AsyncAgentPool — 异步子智能体并行
  5. AsyncEngineFactory — 组件构建
  6. async_prove_round — 端到端异步编排
  7. 并发正确性 — 多路 gather 不竞态
"""
import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


# ═══════════════════════════════════════════════════════════════
# 1. AsyncLeanPool
# ═══════════════════════════════════════════════════════════════

class TestAsyncLeanPool:

    @pytest.mark.asyncio
    async def test_pool_start_fallback(self):
        """Pool 应在无 REPL 时进入 fallback 模式"""
        from engine.async_lean_pool import AsyncLeanPool
        pool = AsyncLeanPool(pool_size=2, project_dir="/tmp")
        ok = await pool.start()
        assert ok is True
        stats = pool.stats()
        assert stats["active_sessions"] == 2
        assert stats["all_fallback"] is True
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_try_tactic_fallback(self):
        """Fallback 模式下 try_tactic 应返回 no_backend 错误"""
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            result = await pool.try_tactic(0, "simp")
            assert result.success is False
            assert result.error_category == "no_backend"

    @pytest.mark.asyncio
    async def test_try_tactics_parallel(self):
        """并行 tactic 应返回与输入等长的结果列表"""
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=2, project_dir="/tmp") as pool:
            tactics = ["simp", "ring", "omega", "linarith"]
            results = await pool.try_tactics_parallel(0, tactics)
            assert len(results) == 4
            for r in results:
                assert hasattr(r, 'success')
                assert hasattr(r, 'tactic')

    @pytest.mark.asyncio
    async def test_verify_complete_with_cache(self):
        """第二次验证相同证明应命中缓存"""
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            r1 = await pool.verify_complete("theorem t : True", ":= trivial")
            r2 = await pool.verify_complete("theorem t : True", ":= trivial")
            # 第二次应该很快 (缓存命中)
            assert pool._compile_cache.hits >= 1

    @pytest.mark.asyncio
    async def test_acquire_release_session(self):
        """获取和释放会话应正确标记 busy"""
        from engine.async_lean_pool import AsyncLeanPool
        pool = AsyncLeanPool(pool_size=1, project_dir="/tmp")
        await pool.start()

        s = await pool._acquire_session()
        assert s.is_busy is True
        await pool._release_session(s)
        assert s.is_busy is False

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_parallel_acquire_no_deadlock(self):
        """多路并行获取不应死锁"""
        from engine.async_lean_pool import AsyncLeanPool
        pool = AsyncLeanPool(pool_size=2, project_dir="/tmp")
        await pool.start()

        async def use_session():
            s = await pool._acquire_session()
            await asyncio.sleep(0.05)
            await pool._release_session(s)

        # 4 个协程争夺 2 个会话, 不应死锁
        await asyncio.wait_for(
            asyncio.gather(*[use_session() for _ in range(4)]),
            timeout=5.0)

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """async with 应自动 start/shutdown"""
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            assert pool._started is True
        assert pool._started is False

    @pytest.mark.asyncio
    async def test_empty_tactics_list(self):
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            results = await pool.try_tactics_parallel(0, [])
            assert results == []


# ═══════════════════════════════════════════════════════════════
# 2. AsyncVerificationScheduler
# ═══════════════════════════════════════════════════════════════

class TestAsyncVerificationScheduler:

    @pytest.mark.asyncio
    async def test_l0_reject_sorry(self):
        """L0 应拒绝含 sorry 的证明"""
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        scheduler = AsyncVerificationScheduler()
        result = await scheduler.verify_tactic(0, "sorry")
        assert result.success is False
        assert result.level_reached == "L0"
        assert result.l0_passed is False

    @pytest.mark.asyncio
    async def test_l0_pass_clean_tactic(self):
        """L0 应放行合法 tactic"""
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            scheduler = AsyncVerificationScheduler(lean_pool=pool)
            result = await scheduler.verify_tactic(0, "simp")
            assert result.l0_passed is True

    @pytest.mark.asyncio
    async def test_verify_complete(self):
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        scheduler = AsyncVerificationScheduler()
        result = await scheduler.verify_complete(
            "theorem t : True", ":= by sorry")
        # sorry should be caught by L0
        assert result.l0_passed is False

    @pytest.mark.asyncio
    async def test_verify_tactics_parallel(self):
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=2, project_dir="/tmp") as pool:
            scheduler = AsyncVerificationScheduler(lean_pool=pool)
            results = await scheduler.verify_tactics_parallel(
                0, ["simp", "sorry", "ring"])
            assert len(results) == 3
            # "sorry" should be L0-rejected
            sorry_result = results[1]
            assert sorry_result.l0_passed is False

    @pytest.mark.asyncio
    async def test_stats(self):
        from engine.async_verification_scheduler import AsyncVerificationScheduler
        scheduler = AsyncVerificationScheduler()
        await scheduler.verify_tactic(0, "sorry")
        await scheduler.verify_tactic(0, "simp")
        s = scheduler.stats()
        assert s["total"] >= 2
        assert s["l0_rejected"] >= 1


# ═══════════════════════════════════════════════════════════════
# 3. AsyncLLMProvider
# ═══════════════════════════════════════════════════════════════

class TestAsyncLLMProvider:

    @pytest.mark.asyncio
    async def test_mock_provider(self):
        from agent.brain.async_llm_provider import AsyncMockProvider
        provider = AsyncMockProvider()
        resp = await provider.generate(system="test", user="prove it")
        assert resp.content
        assert resp.model == "async-mock"
        assert resp.tokens_in > 0

    @pytest.mark.asyncio
    async def test_cached_provider(self):
        from agent.brain.async_llm_provider import (
            AsyncMockProvider, AsyncCachedProvider)
        base = AsyncMockProvider()
        cached = AsyncCachedProvider(base, cache_all=True)

        r1 = await cached.generate(system="s", user="u", temperature=0.5)
        r2 = await cached.generate(system="s", user="u", temperature=0.5)
        assert cached.hits == 1
        assert r2.cached is True

    @pytest.mark.asyncio
    async def test_parallel_llm_calls(self):
        """多路 LLM 调用应真正并行"""
        from agent.brain.async_llm_provider import AsyncMockProvider
        provider = AsyncMockProvider()

        t0 = time.time()
        results = await asyncio.gather(
            *[provider.generate(user=f"task_{i}") for i in range(10)])
        elapsed = time.time() - t0

        assert len(results) == 10
        # 10 个 0.01s 的 mock 调用并行执行应远小于 0.1s (串行)
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_factory(self):
        from agent.brain.async_llm_provider import create_async_provider
        p = create_async_provider({"provider": "mock"})
        r = await p.generate(user="test")
        assert r.content


# ═══════════════════════════════════════════════════════════════
# 4. AsyncAgentPool
# ═══════════════════════════════════════════════════════════════

class TestAsyncAgentPool:

    @pytest.mark.asyncio
    async def test_run_single(self):
        from agent.brain.async_llm_provider import AsyncMockProvider
        from agent.runtime.async_agent_pool import AsyncAgentPool
        from agent.runtime.sub_agent import AgentSpec, AgentTask
        from agent.brain.roles import AgentRole

        pool = AsyncAgentPool(llm=AsyncMockProvider())
        spec = AgentSpec(name="test", role=AgentRole.PROOF_GENERATOR)
        task = AgentTask(description="prove True := trivial")
        result = await pool.run_single(spec, task)
        assert result.agent_name == "test"
        assert result.content

    @pytest.mark.asyncio
    async def test_run_parallel(self):
        from agent.brain.async_llm_provider import AsyncMockProvider
        from agent.runtime.async_agent_pool import AsyncAgentPool
        from agent.runtime.sub_agent import AgentSpec, AgentTask
        from agent.brain.roles import AgentRole

        pool = AsyncAgentPool(llm=AsyncMockProvider(), max_workers=4)
        specs_and_tasks = [
            (AgentSpec(name=f"agent_{i}", role=AgentRole.PROOF_GENERATOR),
             AgentTask(description=f"task_{i}"))
            for i in range(6)
        ]

        t0 = time.time()
        results = await pool.run_parallel(specs_and_tasks)
        elapsed = time.time() - t0

        assert len(results) == 6
        for r in results:
            assert r.agent_name.startswith("agent_")
        # 6 个 mock 调用并行应远小于串行
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_run_then_fuse(self):
        from agent.brain.async_llm_provider import AsyncMockProvider
        from agent.runtime.async_agent_pool import AsyncAgentPool
        from agent.runtime.sub_agent import AgentSpec, AgentTask
        from agent.brain.roles import AgentRole

        pool = AsyncAgentPool(llm=AsyncMockProvider())
        specs_and_tasks = [
            (AgentSpec(name=f"a{i}", role=AgentRole.PROOF_GENERATOR),
             AgentTask(description=f"t{i}"))
            for i in range(3)
        ]
        best, all_results = await pool.run_then_fuse(specs_and_tasks)
        assert best is not None
        assert len(all_results) == 3

    @pytest.mark.asyncio
    async def test_pipeline(self):
        from agent.brain.async_llm_provider import AsyncMockProvider
        from agent.runtime.async_agent_pool import AsyncAgentPool
        from agent.runtime.sub_agent import AgentSpec, AgentTask
        from agent.brain.roles import AgentRole

        pool = AsyncAgentPool(llm=AsyncMockProvider())
        stages = [
            {"spec": AgentSpec(name="planner", role=AgentRole.PROOF_PLANNER),
             "task_template": "Plan a proof"},
            {"spec": AgentSpec(name="generator", role=AgentRole.PROOF_GENERATOR),
             "task_template": "Generate proof based on plan"},
        ]
        result = await pool.run_pipeline(stages)
        assert result.agent_name == "generator"

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """子智能体异常不应导致整体崩溃"""
        from agent.brain.async_llm_provider import AsyncLLMProvider
        from agent.brain.llm_provider import LLMResponse
        from agent.runtime.async_agent_pool import AsyncAgentPool
        from agent.runtime.sub_agent import AgentSpec, AgentTask
        from agent.brain.roles import AgentRole

        class FailingProvider(AsyncLLMProvider):
            @property
            def model_name(self): return "fail"
            async def generate(self, **kwargs):
                raise RuntimeError("API explosion")

        pool = AsyncAgentPool(llm=FailingProvider())
        specs_and_tasks = [
            (AgentSpec(name="crasher", role=AgentRole.PROOF_GENERATOR),
             AgentTask(description="boom"))
        ]
        results = await pool.run_parallel(specs_and_tasks)
        assert len(results) == 1
        assert results[0].error  # Should have error, not crash


# ═══════════════════════════════════════════════════════════════
# 5. AsyncEngineFactory
# ═══════════════════════════════════════════════════════════════

class TestAsyncEngineFactory:

    @pytest.mark.asyncio
    async def test_factory_build(self):
        from engine.async_factory import AsyncEngineFactory
        from agent.brain.async_llm_provider import AsyncMockProvider

        factory = AsyncEngineFactory({"lean_pool_size": 1})
        components = await factory.build(async_llm=AsyncMockProvider())

        assert components.lean_pool is not None
        assert components.scheduler is not None
        assert components.broadcast is not None
        assert components.agent_pool is not None

        await components.close()

    @pytest.mark.asyncio
    async def test_factory_overrides(self):
        from engine.async_factory import AsyncEngineFactory
        from engine.broadcast import BroadcastBus

        custom_bus = BroadcastBus()
        factory = AsyncEngineFactory()
        components = await factory.build(
            overrides={"broadcast": custom_bus})
        assert components.broadcast is custom_bus
        await components.close()


# ═══════════════════════════════════════════════════════════════
# 6. End-to-end async_prove_round
# ═══════════════════════════════════════════════════════════════

class TestAsyncProveRound:

    @pytest.mark.asyncio
    async def test_e2e_prove_round(self):
        """端到端: 异步构建 → 异步证明 → 结果排序"""
        from engine.async_factory import AsyncEngineFactory, async_prove_round
        from agent.brain.async_llm_provider import AsyncMockProvider
        from prover.models import BenchmarkProblem

        factory = AsyncEngineFactory({"lean_pool_size": 1})
        components = await factory.build(async_llm=AsyncMockProvider())

        problem = BenchmarkProblem(
            problem_id="test_1",
            name="test_true",
            theorem_statement="theorem test_true : True",
            natural_language="Prove True",
            difficulty="easy")

        results = await async_prove_round(problem, components)
        assert len(results) >= 3  # At least 3 directions
        for r in results:
            assert hasattr(r, 'confidence')
            assert hasattr(r, 'proof_code')

        await components.close()

    @pytest.mark.asyncio
    async def test_sync_entry_point(self):
        """run_async_prove_round 同步入口"""
        from engine.async_factory import (
            AsyncEngineFactory, async_prove_round as _apr,
            AsyncEngineComponents)
        from agent.brain.async_llm_provider import AsyncMockProvider
        from prover.models import BenchmarkProblem

        # 需要在事件循环外调用, 但 pytest-asyncio 已经有事件循环
        # 所以这里直接测试 async 版本
        factory = AsyncEngineFactory({"lean_pool_size": 1})
        components = await factory.build(async_llm=AsyncMockProvider())

        problem = BenchmarkProblem(
            problem_id="t2", name="t2",
            theorem_statement="theorem t2 : 1 = 1",
            natural_language="", difficulty="easy")

        results = await _apr(problem, components)
        assert isinstance(results, list)
        await components.close()


# ═══════════════════════════════════════════════════════════════
# 7. 并发正确性
# ═══════════════════════════════════════════════════════════════

class TestConcurrencyCorrectness:

    @pytest.mark.asyncio
    async def test_no_session_double_acquire(self):
        """两个协程不应同时获得同一个 session"""
        from engine.async_lean_pool import AsyncLeanPool
        pool = AsyncLeanPool(pool_size=2, project_dir="/tmp")
        await pool.start()

        acquired_ids = []
        lock = asyncio.Lock()

        async def grab():
            s = await pool._acquire_session()
            async with lock:
                # 在释放前检查没有重复
                assert s.session_id not in acquired_ids, \
                    f"Session {s.session_id} double-acquired!"
                acquired_ids.append(s.session_id)
            await asyncio.sleep(0.05)
            async with lock:
                acquired_ids.remove(s.session_id)
            await pool._release_session(s)

        await asyncio.gather(*[grab() for _ in range(8)])
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_gather_preserves_order(self):
        """asyncio.gather 应保持结果顺序"""
        from engine.async_lean_pool import AsyncLeanPool
        async with AsyncLeanPool(pool_size=2, project_dir="/tmp") as pool:
            tactics = [f"tactic_{i}" for i in range(5)]
            results = await pool.try_tactics_parallel(0, tactics)
            for i, r in enumerate(results):
                assert r.tactic == f"tactic_{i}"

    @pytest.mark.asyncio
    async def test_mixed_parallel_workload(self):
        """LLM 调用和 REPL 验证应在同一事件循环中交替执行"""
        from engine.async_lean_pool import AsyncLeanPool
        from agent.brain.async_llm_provider import AsyncMockProvider

        provider = AsyncMockProvider()

        async with AsyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            # 同时发起 LLM 调用 + REPL 验证
            llm_task = provider.generate(user="prove something")
            repl_task = pool.try_tactic(0, "simp")

            llm_result, repl_result = await asyncio.gather(
                llm_task, repl_task)

            assert llm_result.content
            assert hasattr(repl_result, 'success')

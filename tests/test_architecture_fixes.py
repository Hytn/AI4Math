"""tests/test_architecture_fixes.py — 架构修复的回归测试

覆盖 Phase 0-5 的所有新增/修改模块:
  - assemble_code 5种拼接模式
  - Transport 协议层 (MockTransport, SyncTransportAdapter)
  - SyncLeanPool 包装器
  - MetricsCollector / StructuredLogger / timed 装饰器
  - DirectionPlanner
  - ProofPipeline (init/stages)
  - ConfidenceEstimator.refine_confidence (统一归口)
  - AgentResult.is_error
  - ContextWindow._compress Phase 3 改进
  - BroadcastMessage 工厂 freeze
  - ProofSession _trace_path 循环检测
  - Config env var overrides
  - Engine API protocols
"""
import asyncio
import os
import pytest
import time


# ═══════════════════════════════════════════════════════════════
# Phase 0: assemble_code
# ═══════════════════════════════════════════════════════════════

class TestAssembleCode:
    def test_theorem_with_proof_by(self):
        from engine._core import assemble_code
        result = assemble_code("theorem t : True", "by trivial")
        assert ":= by" in result
        assert "trivial" in result

    def test_theorem_with_proof_assign(self):
        from engine._core import assemble_code
        result = assemble_code("theorem t : True", ":= by trivial")
        assert "theorem t : True := by trivial" in result

    def test_theorem_already_has_assign(self):
        from engine._core import assemble_code
        result = assemble_code("theorem t : True := by trivial", "extra")
        # Should NOT append proof since theorem already has :=
        assert "extra" not in result
        assert ":= by trivial" in result

    def test_theorem_with_bare_tactic(self):
        from engine._core import assemble_code
        result = assemble_code("theorem t : 1+1=2", "norm_num")
        assert ":= by" in result
        assert "norm_num" in result

    def test_empty_proof(self):
        from engine._core import assemble_code
        result = assemble_code("theorem t : True", "")
        assert "theorem t : True" in result
        assert ":=" not in result

    def test_empty_theorem(self):
        from engine._core import assemble_code
        result = assemble_code("", "by trivial")
        assert "by trivial" in result

    def test_custom_preamble(self):
        from engine._core import assemble_code
        result = assemble_code("theorem t : True", "by trivial",
                               preamble="import Std")
        assert result.startswith("import Std")
        assert "import Mathlib" not in result

    def test_no_double_assign(self):
        """Regression: theorem := by ... proof := by ... should not happen"""
        from engine._core import assemble_code
        result = assemble_code("theorem t : True := by sorry", ":= by trivial")
        assert result.count(":=") == 1


# ═══════════════════════════════════════════════════════════════
# Phase 1: Transport
# ═══════════════════════════════════════════════════════════════

class TestMockTransport:
    def test_basic_send_receive(self):
        from engine.transport import MockTransport

        responses = [
            {"env": 1, "messages": [], "goals": []},
            {"env": 2, "messages": [], "goals": ["⊢ True"]},
        ]
        transport = MockTransport(responses)

        async def _run():
            await transport.start()
            assert transport.is_alive
            assert not transport.is_fallback

            r1 = await transport.send({"cmd": "import Mathlib", "env": 0})
            assert r1["env"] == 1

            r2 = await transport.send({"cmd": "theorem t : True := by", "env": 1})
            assert r2["goals"] == ["⊢ True"]

            # Beyond provided responses → default
            r3 = await transport.send({"cmd": "trivial", "env": 2})
            assert "env" in r3

            assert len(transport._sent_commands) == 3
            await transport.close()
            assert not transport.is_alive

        asyncio.run(_run())

    def test_fallback_transport(self):
        from engine.transport import FallbackTransport

        async def _run():
            t = FallbackTransport()
            await t.start()
            assert t.is_alive
            assert t.is_fallback
            assert await t.send({"cmd": "test"}) is None
            await t.close()

        asyncio.run(_run())


class TestSyncTransportAdapter:
    def test_sync_wrapper(self):
        from engine.transport import MockTransport, SyncTransportAdapter

        transport = MockTransport([{"env": 1, "messages": [], "goals": []}])
        sync = SyncTransportAdapter(transport)

        assert sync.start()
        assert sync.is_alive

        resp = sync.send({"cmd": "test", "env": 0})
        assert resp["env"] == 1

        sync.close()
        assert not sync.is_alive


# ═══════════════════════════════════════════════════════════════
# Phase 1: SyncLeanPool
# ═══════════════════════════════════════════════════════════════

class TestSyncLeanPool:
    def test_basic_lifecycle(self):
        from engine.async_lean_pool import SyncLeanPool

        pool = SyncLeanPool(pool_size=1, project_dir="/tmp")
        pool.start()

        stats = pool.stats()
        assert stats["active_sessions"] >= 1

        # Should work in fallback mode (no lean installed)
        result = pool.try_tactic(0, "simp")
        assert not result.success  # fallback

        pool.shutdown()

    def test_context_manager(self):
        from engine.async_lean_pool import SyncLeanPool

        with SyncLeanPool(pool_size=1, project_dir="/tmp") as pool:
            stats = pool.stats()
            assert "active_sessions" in stats


# ═══════════════════════════════════════════════════════════════
# Phase 2: ConfidenceEstimator.refine_confidence
# ═══════════════════════════════════════════════════════════════

class TestRefineConfidence:
    def _make_result(self, confidence=0.5):
        from agent.runtime.sub_agent import AgentResult
        from agent.brain.roles import AgentRole
        return AgentResult(
            agent_name="test", role=AgentRole.PROOF_GENERATOR,
            content="", confidence=confidence)

    def test_l0_rejected(self):
        from agent.strategy.confidence_estimator import ConfidenceEstimator
        r = self._make_result(0.5)
        c = ConfidenceEstimator.refine_confidence(r, l0_passed=False)
        assert c <= 0.15

    def test_l2_passed(self):
        from agent.strategy.confidence_estimator import ConfidenceEstimator
        r = self._make_result(0.3)
        c = ConfidenceEstimator.refine_confidence(r, l2_passed=True)
        assert c == 0.95

    def test_l1_passed(self):
        from agent.strategy.confidence_estimator import ConfidenceEstimator
        r = self._make_result(0.3)
        c = ConfidenceEstimator.refine_confidence(r, l1_passed=True)
        assert c >= 0.80

    def test_backward_compat_via_subagent(self):
        """SubAgent.refine_confidence should delegate to ConfidenceEstimator"""
        from agent.runtime.sub_agent import SubAgent
        r = self._make_result(0.3)
        c = SubAgent.refine_confidence(r, l2_passed=True)
        assert c == 0.95


# ═══════════════════════════════════════════════════════════════
# Phase 2: AgentResult.is_error
# ═══════════════════════════════════════════════════════════════

class TestAgentResultIsError:
    def test_normal_result(self):
        from agent.runtime.sub_agent import AgentResult
        from agent.brain.roles import AgentRole
        r = AgentResult(agent_name="t", role=AgentRole.PROOF_GENERATOR,
                        content="proof")
        assert not r.is_error

    def test_error_result(self):
        from agent.runtime.sub_agent import AgentResult
        from agent.brain.roles import AgentRole
        r = AgentResult(agent_name="t", role=AgentRole.PROOF_GENERATOR,
                        content="", error="timeout")
        assert r.is_error


# ═══════════════════════════════════════════════════════════════
# Phase 2: DirectionPlanner
# ═══════════════════════════════════════════════════════════════

class TestDirectionPlanner:
    def test_basic_plan(self):
        from agent.strategy.direction_planner import DirectionPlanner
        from prover.models import BenchmarkProblem

        planner = DirectionPlanner()
        problem = BenchmarkProblem(
            problem_id="t1", name="test",
            theorem_statement="theorem t : True")

        directions = planner.plan(problem)
        assert len(directions) >= 3
        names = [d.name for d in directions]
        assert "automation" in names
        assert "structured" in names
        assert "alternative" in names

    def test_repair_direction_with_history(self):
        from agent.strategy.direction_planner import DirectionPlanner
        from prover.models import BenchmarkProblem

        planner = DirectionPlanner()
        problem = BenchmarkProblem(
            problem_id="t1", name="test",
            theorem_statement="theorem t : True")

        history = [
            {"errors": [{"message": "type mismatch"}]},
            {"errors": [{"message": "unknown identifier"}]},
        ]
        directions = planner.plan(problem, attempt_history=history)
        names = [d.name for d in directions]
        assert "repair_rethink" in names

    def test_nat_sub_classification(self):
        from agent.strategy.direction_planner import DirectionPlanner
        from prover.models import BenchmarkProblem

        planner = DirectionPlanner()
        problem = BenchmarkProblem(
            problem_id="t1", name="test",
            theorem_statement="theorem t : n - m + m = n")

        directions = planner.plan(
            problem, classification={"has_nat_sub": True})
        structured = [d for d in directions if d.name == "structured"][0]
        assert "subtraction" in structured.strategic_hint.lower()

    def test_build_direction_prompt(self):
        from agent.strategy.direction_planner import (
            DirectionPlanner, ProofDirection, build_direction_prompt)
        from agent.brain.roles import AgentRole
        from prover.models import BenchmarkProblem

        d = ProofDirection(name="test", role=AgentRole.PROOF_GENERATOR,
                           strategic_hint="Use omega")
        p = BenchmarkProblem(problem_id="t", name="t",
                             theorem_statement="theorem t : 1+1=2")

        prompt = build_direction_prompt(d, p)
        assert "1+1=2" in prompt
        assert "Use omega" in prompt
        assert "sorry" in prompt.lower()


# ═══════════════════════════════════════════════════════════════
# Phase 3: ProofSession._trace_path cycle detection
# ═══════════════════════════════════════════════════════════════

class TestTracePathCycleDetection:
    def test_no_infinite_loop_on_self_reference(self):
        from engine.proof_session import ProofSession, ProofSessionState, EnvNode

        # Simulate fallback mode: root and theorem both env_id=0
        root = EnvNode(env_id=0, parent_env_id=-1, depth=0)
        state = ProofSessionState(
            theorem="test", root_env_id=0,
            theorem_env_id=0, current_env_id=0,
            nodes={0: root})

        session = ProofSession(state, pool=None)
        # Should not hang — cycle detection should break
        path = session._trace_path(0)
        assert len(path) <= 2  # At most root

    def test_normal_path(self):
        from engine.proof_session import ProofSession, ProofSessionState, EnvNode

        nodes = {
            0: EnvNode(env_id=0, parent_env_id=-1, depth=0),
            1: EnvNode(env_id=1, parent_env_id=0, tactic="intro", depth=1),
            2: EnvNode(env_id=2, parent_env_id=1, tactic="simp", depth=2),
        }
        state = ProofSessionState(
            theorem="test", root_env_id=0,
            theorem_env_id=1, current_env_id=2, nodes=nodes)

        session = ProofSession(state, pool=None)
        path = session._trace_path(2)
        assert path == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════
# Phase 4: Observability
# ═══════════════════════════════════════════════════════════════

class TestMetricsCollector:
    def test_timer(self):
        from engine.observability import MetricsCollector

        m = MetricsCollector()
        with m.timer("test_op"):
            time.sleep(0.01)

        snap = m.snapshot()
        assert "test_op" in snap
        assert snap["test_op"]["count"] == 1
        assert snap["test_op"]["min"] >= 5  # at least 5ms

    def test_timer_with_labels(self):
        from engine.observability import MetricsCollector

        m = MetricsCollector()
        with m.timer("verify", level="L1"):
            pass
        with m.timer("verify", level="L2"):
            pass

        snap = m.snapshot()
        assert "verify{level=L1}" in snap
        assert "verify{level=L2}" in snap

    def test_counter(self):
        from engine.observability import MetricsCollector

        m = MetricsCollector()
        m.increment("requests")
        m.increment("requests")
        m.increment("requests", delta=3)

        snap = m.snapshot()
        assert snap["requests"]["value"] == 5

    def test_gauge(self):
        from engine.observability import MetricsCollector

        m = MetricsCollector()
        m.set_gauge("pool_size", 4)
        m.set_gauge("pool_size", 6)

        snap = m.snapshot()
        assert snap["pool_size"]["value"] == 6

    def test_disabled(self):
        from engine.observability import MetricsCollector

        m = MetricsCollector(enabled=False)
        with m.timer("noop"):
            pass
        m.increment("noop")
        assert m.snapshot() == {}

    def test_reset(self):
        from engine.observability import MetricsCollector

        m = MetricsCollector()
        m.increment("x")
        m.reset()
        assert m.snapshot() == {}


class TestTimedDecorator:
    def test_sync_function(self):
        from engine.observability import MetricsCollector, timed
        import engine.observability as obs

        original = obs.metrics
        obs.metrics = MetricsCollector()

        @timed("my_func")
        def slow_func():
            time.sleep(0.01)
            return 42

        result = slow_func()
        assert result == 42

        snap = obs.metrics.snapshot()
        assert "my_func" in snap
        assert snap["my_func"]["count"] == 1

        obs.metrics = original


class TestStructuredLogger:
    def test_bind_creates_new_logger(self):
        from engine.observability import StructuredLogger

        base = StructuredLogger("test")
        bound = base.bind(problem_id="p1")

        assert "problem_id" not in base._context
        assert bound._context["problem_id"] == "p1"

    def test_format(self):
        from engine.observability import StructuredLogger

        slog = StructuredLogger("test", session=1)
        msg = slog._format("tactic_ok", tactic="simp", ms=42)
        assert "tactic_ok" in msg
        assert "session=1" in msg
        assert "tactic=simp" in msg


# ═══════════════════════════════════════════════════════════════
# Phase 5: Config env var overrides
# ═══════════════════════════════════════════════════════════════

class TestConfigEnvOverrides:
    def test_env_override(self):
        from config.schema import load_config

        os.environ["APE_ENGINE__LEAN_POOL_SIZE"] = "16"
        try:
            config = load_config("nonexistent.yaml")
            assert config.get("engine", {}).get("lean_pool_size") == 16
        finally:
            del os.environ["APE_ENGINE__LEAN_POOL_SIZE"]

    def test_env_bool_coercion(self):
        from config.schema import _coerce_value
        assert _coerce_value("true") is True
        assert _coerce_value("false") is False
        assert _coerce_value("42") == 42
        assert _coerce_value("3.14") == 3.14
        assert _coerce_value("hello") == "hello"
        assert _coerce_value("null") is None

    def test_nested_env_override(self):
        from config.schema import load_config

        os.environ["APE_AGENT__BRAIN__MODEL"] = "claude-opus-4-20250514"
        try:
            config = load_config("nonexistent.yaml")
            assert config["agent"]["brain"]["model"] == "claude-opus-4-20250514"
        finally:
            del os.environ["APE_AGENT__BRAIN__MODEL"]


# ═══════════════════════════════════════════════════════════════
# Phase 2: BroadcastMessage freeze safety
# ═══════════════════════════════════════════════════════════════

class TestBroadcastMessageFreeze:
    def test_factory_methods_freeze_structured(self):
        from engine.broadcast import BroadcastMessage
        from types import MappingProxyType

        msg = BroadcastMessage.negative(
            source="test", tactic="simp",
            error_category="type_mismatch", reason="nope")

        # structured should be frozen (MappingProxyType)
        assert isinstance(msg.structured, MappingProxyType)

        # Should not be mutable
        with pytest.raises(TypeError):
            msg.structured["new_key"] = "value"

    def test_partial_proof_freeze(self):
        from engine.broadcast import BroadcastMessage

        msg = BroadcastMessage.partial_proof(
            source="test", proof_so_far="by simp",
            remaining_goals=["⊢ True"], goals_closed=1)

        # remaining_goals should be frozen to tuple
        assert isinstance(msg.structured["remaining_goals"], tuple)


# ═══════════════════════════════════════════════════════════════
# Phase 2: Engine API protocols
# ═══════════════════════════════════════════════════════════════

class TestEngineProtocols:
    def test_sync_lean_pool_satisfies_protocol(self):
        """SyncLeanPool should satisfy PoolProtocol"""
        from engine.api.protocols import PoolProtocol
        from engine.async_lean_pool import SyncLeanPool

        pool = SyncLeanPool(pool_size=1, project_dir="/tmp")
        # Check key methods exist (duck typing)
        assert hasattr(pool, 'try_tactic')
        assert hasattr(pool, 'verify_complete')
        assert hasattr(pool, 'try_tactics_parallel')
        assert hasattr(pool, 'stats')
        assert hasattr(pool, 'shutdown')


# ═══════════════════════════════════════════════════════════════
# Phase 0: ContextWindow compress improvement
# ═══════════════════════════════════════════════════════════════

class TestContextWindowCompress:
    def test_phase3_retains_more_info(self):
        from agent.context.context_window import ContextWindow

        ctx = ContextWindow(max_tokens=500, compress_threshold=0.3)

        # Add many attempt entries to trigger Phase 3
        for i in range(8):
            ctx.add_entry(f"attempt_{i}",
                          f"Failed attempt {i}\nerror_category: type_mismatch\n"
                          f"Details: some long error message here",
                          priority=0.3, category="attempt")

        # Force compression
        ctx._compress()

        # Check that compressed history exists and has useful info
        compressed = ctx.get_entry("_compressed_history")
        if compressed:
            assert "type_mismatch" in compressed.content or "Failed" in compressed.content


# ═══════════════════════════════════════════════════════════════
# Phase 0: Overflow session cleanup
# ═══════════════════════════════════════════════════════════════

class TestOverflowCleanup:
    def test_overflow_flag_exists(self):
        from engine.lean_pool import _SessionState
        state = _SessionState()
        assert hasattr(state, 'is_overflow')
        assert state.is_overflow is False

    def test_async_session_overflow_flag(self):
        from engine.async_lean_pool import AsyncLeanSession
        session = AsyncLeanSession(session_id=0)
        assert hasattr(session, '_is_overflow')
        assert session._is_overflow is False

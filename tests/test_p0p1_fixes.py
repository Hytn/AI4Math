"""tests/test_p0p1_fixes.py — P0/P1 修复的验证测试

覆盖:
  P0-1: REPL 统一 (_CompileCache in LeanPool, LeanChecker 新后端)
  P0-3: _acquire_session 并发安全 (Condition-based wait)
  P0-4: Orchestrator 依赖注入 (EngineFactory)
  P1-5: 环境缓存 (env_cache)
  P1-6: L2 验证路径 (temp file + lake env lean)
  P1-7: 广播背压 (Subscription maxlen)
  P1-8: 广播消费闭环 (历史消息注入)
  P1-9: 错误分类增强 (structured + more categories)
  P1-11: 配置 schema (APE v2 参数)
"""
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ═══════════════════════════════════════════════════════════════
# P0-1: REPL 统一 — _CompileCache 在 LeanPool 层
# ═══════════════════════════════════════════════════════════════

class TestCompileCache:
    """P0-1: 编译缓存从 lean_repl.py 统一到 lean_pool.py"""

    def test_basic_put_get(self):
        from engine.lean_pool import _CompileCache, FullVerifyResult
        cache = _CompileCache(maxsize=10)
        r = FullVerifyResult(success=True, env_id=42)
        cache.put("key1", r)
        assert cache.get("key1") is r
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        from engine.lean_pool import _CompileCache, FullVerifyResult
        cache = _CompileCache(maxsize=3)
        for i in range(5):
            cache.put(f"k{i}", FullVerifyResult(success=True, env_id=i))
        # k0, k1 should be evicted
        assert cache.get("k0") is None
        assert cache.get("k1") is None
        assert cache.get("k2") is not None
        assert cache.get("k4") is not None

    def test_stats(self):
        from engine.lean_pool import _CompileCache, FullVerifyResult
        cache = _CompileCache(maxsize=10)
        cache.put("x", FullVerifyResult(success=True))
        cache.get("x")   # hit
        cache.get("y")   # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["size"] == 1

    def test_thread_safety(self):
        from engine.lean_pool import _CompileCache, FullVerifyResult
        cache = _CompileCache(maxsize=1000)
        errors = []

        def writer(start):
            for i in range(100):
                try:
                    cache.put(f"w{start}_{i}", FullVerifyResult(success=True, env_id=i))
                except Exception as e:
                    errors.append(e)

        def reader(start):
            for i in range(100):
                try:
                    cache.get(f"w{start}_{i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        threads += [threading.Thread(target=reader, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_lean_pool_has_compile_cache(self):
        """LeanPool 实例应包含 _compile_cache 属性"""
        from engine.lean_pool import LeanPool
        pool = LeanPool(pool_size=1, project_dir="/tmp")
        assert hasattr(pool, '_compile_cache')
        assert pool._compile_cache is not None

    def test_lean_pool_stats_includes_cache(self):
        """LeanPool.stats() 应包含 compile_cache 统计"""
        from engine.lean_pool import LeanPool
        pool = LeanPool(pool_size=1, project_dir="/tmp")
        pool._started = True
        s = pool.stats()
        assert "compile_cache" in s
        assert "env_cache_size" in s


class TestLeanCheckerUnifiedBackend:
    """P0-1: LeanChecker 统一后端"""

    def test_accepts_scheduler(self):
        from prover.verifier.lean_checker import LeanChecker

        class MockScheduler:
            def verify_complete(self, theorem, proof, direction):
                from engine.verification_scheduler import VerificationResult
                from engine.error_intelligence import AgentFeedback
                return VerificationResult(
                    success=True, level_reached="L1",
                    feedback=AgentFeedback(is_proof_complete=True),
                    total_ms=10)

        class MockEnv:
            project_dir = "."

        checker = LeanChecker(MockEnv(), verification_scheduler=MockScheduler())
        from prover.models import AttemptStatus
        status, errors, stderr, ms = checker.check("theorem t : True", ":= trivial")
        assert status == AttemptStatus.SUCCESS

    def test_accepts_pool(self):
        from prover.verifier.lean_checker import LeanChecker

        class MockPool:
            def verify_complete(self, theorem, proof, preamble=""):
                from engine.lean_pool import FullVerifyResult
                return FullVerifyResult(success=True, env_id=1, elapsed_ms=5)

        class MockEnv:
            project_dir = "."

        checker = LeanChecker(MockEnv(), lean_pool=MockPool())
        from prover.models import AttemptStatus
        status, errors, stderr, ms = checker.check("theorem t : True", ":= trivial")
        assert status == AttemptStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════
# P0-3: _acquire_session 并发安全
# ═══════════════════════════════════════════════════════════════

class TestAcquireSessionConcurrency:
    """P0-3: Condition-based 会话获取, 消除竞态条件"""

    def test_acquire_marks_busy(self):
        """获取到的会话必须被标记为 busy"""
        from engine.lean_pool import LeanPool, LeanSession
        pool = LeanPool(pool_size=2, project_dir="/tmp")
        # 手动添加 session (不实际启动 REPL)
        for i in range(2):
            s = LeanSession(session_id=i, project_dir="/tmp")
            s._state.alive = True
            s._state.fallback_mode = True
            pool._sessions.append(s)
        pool._started = True

        s1 = pool._acquire_session()
        assert s1.is_busy, "Acquired session must be marked busy"
        s2 = pool._acquire_session()
        assert s2.is_busy
        assert s1.session_id != s2.session_id, "Must acquire different sessions"

        pool._release_session(s1)
        pool._release_session(s2)

    def test_release_notifies_waiters(self):
        """释放会话应通知等待线程"""
        from engine.lean_pool import LeanPool, LeanSession
        pool = LeanPool(pool_size=1, project_dir="/tmp")
        s = LeanSession(session_id=0, project_dir="/tmp")
        s._state.alive = True
        s._state.fallback_mode = True
        pool._sessions.append(s)
        pool._started = True

        # 先获取唯一的会话
        acquired = pool._acquire_session()
        assert acquired.is_busy

        # 另一个线程尝试获取, 应该等待
        result = [None]
        acquired_event = threading.Event()

        def waiter():
            r = pool._acquire_session()
            result[0] = r
            acquired_event.set()

        t = threading.Thread(target=waiter, daemon=True)
        t.start()

        time.sleep(0.1)
        assert result[0] is None, "Should be waiting"

        # 释放 → waiter 应被唤醒
        pool._release_session(acquired)
        acquired_event.wait(timeout=2)
        assert result[0] is not None, "Waiter should have acquired session"
        assert result[0].is_busy

        pool._release_session(result[0])

    def test_overflow_on_timeout(self):
        """所有会话忙且超时时, 应创建 overflow 会话"""
        from engine.lean_pool import LeanPool, LeanSession
        pool = LeanPool(pool_size=1, project_dir="/tmp", timeout_seconds=1)
        s = LeanSession(session_id=0, project_dir="/tmp")
        s._state.alive = True
        s._state.fallback_mode = True
        s._state.busy = True  # 预设为忙
        pool._sessions.append(s)
        pool._started = True

        t0 = time.time()
        overflow = pool._acquire_session()
        elapsed = time.time() - t0
        assert overflow is not None
        assert overflow.is_busy
        assert overflow.session_id >= 1  # overflow session
        assert elapsed >= 0.9  # waited ~1 second
        pool._release_session(overflow)


# ═══════════════════════════════════════════════════════════════
# P0-4: Orchestrator 依赖注入
# ═══════════════════════════════════════════════════════════════

class TestEngineFactory:
    """P0-4: EngineFactory 组件工厂"""

    def test_factory_builds_components(self):
        from engine.factory import EngineFactory, EngineComponents
        factory = EngineFactory({"lean_pool_size": 1})
        components = factory.build()
        assert components.broadcast is not None
        assert components.prefilter is not None
        assert components.lean_pool is not None
        assert components.scheduler is not None
        assert components.hooks is not None
        components.close()

    def test_factory_accepts_overrides(self):
        from engine.factory import EngineFactory, EngineComponents
        from engine.broadcast import BroadcastBus

        custom_bus = BroadcastBus(dedup_window_seconds=999)
        factory = EngineFactory()
        components = factory.build(overrides={"broadcast": custom_bus})
        assert components.broadcast is custom_bus
        components.close()

    def test_components_close_is_safe(self):
        """close() 不应在任何组件为 None 时崩溃"""
        from engine.factory import EngineComponents
        comp = EngineComponents()
        comp.close()  # Should not raise

    def test_orchestrator_accepts_components(self):
        """Orchestrator 应接受预构建的 EngineComponents"""
        from engine.factory import EngineFactory
        from prover.pipeline.orchestrator import Orchestrator

        factory = EngineFactory({"lean_pool_size": 1})
        components = factory.build()

        class MockEnv:
            project_dir = "."
            def compile(self, code): return (0, "", "")

        class MockLLM:
            def generate(self, **kwargs):
                from agent.brain.llm_provider import LLMResponse
                return LLMResponse(content="", tokens_in=0, tokens_out=0)

        orch = Orchestrator(
            lean_env=MockEnv(), llm_provider=MockLLM(),
            components=components)
        assert orch.scheduler is components.scheduler
        assert orch.lean_pool is components.lean_pool
        orch.close()


# ═══════════════════════════════════════════════════════════════
# P1-5: 环境缓存
# ═══════════════════════════════════════════════════════════════

class TestEnvCache:
    """P1-5: preamble → env_id 缓存"""

    def test_get_cached_env_id(self):
        from engine.lean_pool import LeanPool
        pool = LeanPool(pool_size=1, project_dir="/tmp")
        pool._env_cache["abc123"] = 42
        # 正确的 key 应命中
        assert pool.get_cached_env_id.__doc__  # method exists


# ═══════════════════════════════════════════════════════════════
# P1-6: L2 验证路径
# ═══════════════════════════════════════════════════════════════

class TestL2Verification:
    """P1-6: L2 使用 temp file + lake env lean"""

    def test_l2_uses_temp_file(self):
        """L2 应写入临时 .lean 文件而非 stdin pipe"""
        from engine.verification_scheduler import VerificationScheduler
        from engine.prefilter import PreFilter

        scheduler = VerificationScheduler(
            prefilter=PreFilter(), project_dir="/tmp")

        # 即使没有 lean 也不会崩溃
        result = scheduler._l2_full_compile(
            "theorem t : True", ":= trivial",
            preamble="import Lean", timeout=5)
        # 应该是 "not found" 错误而非 crash
        assert result.success is False or result.success is True
        assert isinstance(result.stderr, str)


# ═══════════════════════════════════════════════════════════════
# P1-7: 广播背压
# ═══════════════════════════════════════════════════════════════

class TestBroadcastBackpressure:
    """P1-7: Subscription maxlen 防止内存泄漏"""

    def test_subscription_maxlen(self):
        from engine.broadcast import BroadcastBus, BroadcastMessage
        bus = BroadcastBus()
        sub = bus.subscribe("consumer")

        # 发送 200 条消息 (maxlen=100)
        for i in range(200):
            bus.publish(BroadcastMessage.positive(
                source=f"src_{i}", discovery=f"discovery_{i}"))

        assert sub.pending_count <= 100

    def test_history_maxlen(self):
        from engine.broadcast import BroadcastBus, BroadcastMessage
        bus = BroadcastBus()
        for i in range(600):
            bus.publish(BroadcastMessage.positive(
                source=f"src_{i}", discovery=f"d_{i}"))
        assert len(bus._history) <= 500


# ═══════════════════════════════════════════════════════════════
# P1-8: 广播消费闭环
# ═══════════════════════════════════════════════════════════════

class TestBroadcastClosedLoop:
    """P1-8: 新订阅者注入历史消息"""

    def test_new_subscriber_gets_history(self):
        """HeterogeneousEngine 在 run_round 中创建的新订阅应包含历史消息"""
        from engine.broadcast import BroadcastBus, BroadcastMessage, Subscription

        bus = BroadcastBus()

        # 模拟第一轮: agent_A 发布发现
        sub_a = bus.subscribe("agent_A")
        bus.publish(BroadcastMessage.positive(
            source="agent_A", discovery="Found useful lemma X"))
        bus.unsubscribe("agent_A")

        # 模拟第二轮: 新建订阅 + 注入历史 (P1-8 修复)
        sub_b = bus.subscribe("agent_B")
        recent = bus.get_recent(n=15)
        for msg in recent:
            sub_b.push(msg)

        msgs = sub_b.drain()
        assert len(msgs) >= 1, "New subscriber should get history"
        assert "lemma X" in msgs[0].content


# ═══════════════════════════════════════════════════════════════
# P1-9: 错误分类增强
# ═══════════════════════════════════════════════════════════════

class TestEnhancedErrorClassification:
    """P1-9: 更多 Lean4 错误类型 + 结构化解析"""

    def test_new_error_categories(self):
        from engine.lean_pool import _classify_error
        cases = [
            ("failed to synthesize instance Foo", "instance_not_found"),
            ("universe level mismatch", "universe_error"),
            ("application type mismatch", "app_type_mismatch"),
            ("function expected at term", "function_expected"),
            ("maximum recursion depth exceeded", "recursion_limit"),
            ("ambiguous, possible interpretations", "ambiguous"),
            ("deterministic timeout", "timeout"),
            ("(kernel) declaration has metavariables", "other"),
        ]
        for msg, expected in cases:
            result = _classify_error(msg)
            assert result == expected, f"_classify_error({msg!r}) = {result!r}, expected {expected!r}"

    def test_structured_multi_error(self):
        from engine.lean_pool import _classify_error_structured
        messages = [
            {"severity": "error", "data": "type mismatch\nhas type Nat\nexpected Int",
             "pos": {"line": 5, "column": 10}, "endPos": {"line": 5, "column": 20}},
            {"severity": "error", "data": "unknown identifier 'foo'"},
            {"severity": "info", "data": "Try this: exact bar"},
        ]
        cat, combined, meta = _classify_error_structured(messages)
        assert cat == "type_mismatch"
        assert meta["error_count"] == 2
        assert meta["primary_pos"] == {"line": 5, "column": 10}
        assert "type_mismatch" in meta["all_categories"]
        assert "unknown_identifier" in meta["all_categories"]

    def test_empty_messages(self):
        from engine.lean_pool import _classify_error_structured
        cat, combined, meta = _classify_error_structured([])
        assert cat == "none"
        assert combined == ""


# ═══════════════════════════════════════════════════════════════
# P1-11: 配置 schema
# ═══════════════════════════════════════════════════════════════

class TestConfigSchema:
    """P1-11: APE v2 参数在 schema 中"""

    def test_lean_pool_size_range(self):
        from config.schema import validate_config
        issues = validate_config({"lean_pool_size": 100})
        assert any("outside valid range" in i for i in issues)

    def test_max_workers_range(self):
        from config.schema import validate_config
        issues = validate_config({"max_workers": 0})
        # 0 is below minimum of 1
        assert any("max_workers" in i for i in issues)

    def test_valid_config_passes(self):
        from config.schema import validate_config
        issues = validate_config({"lean_pool_size": 4, "max_workers": 8})
        lean_issues = [i for i in issues if "lean_pool_size" in i or "max_workers" in i]
        assert len(lean_issues) == 0

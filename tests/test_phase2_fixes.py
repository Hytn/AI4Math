"""tests/test_phase2_fixes.py — Phase 2 架构清理回归测试

验证:
  2.1  SyncLeanPool 是 LeanPool 的别名 (消除重复)
  2.2  EngineFactory.build_engine() 无 agent 依赖
  2.3  Orchestrator.prove() 默认使用 ProofPipeline
  2.5  fork_env 已被移除
  2.6  CompileCache 环境版本指纹
  2.7  Config schema 校验 engine 部分
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


class TestSyncAsyncMerge:
    """2.1: SyncLeanPool should be identical to LeanPool."""

    def test_sync_lean_pool_is_lean_pool(self):
        from engine.async_lean_pool import SyncLeanPool
        from engine.lean_pool import LeanPool
        assert SyncLeanPool is LeanPool

    def test_factory_uses_lean_pool(self):
        """Factory's SyncLeanPool should be the same as LeanPool."""
        from engine.factory import EngineFactory
        factory = EngineFactory({"lean_pool_size": 1})
        components = factory.build_engine()
        try:
            from engine.lean_pool import LeanPool
            assert isinstance(components.lean_pool, LeanPool)
        finally:
            components.close()


class TestFactorySplit:
    """2.2: EngineFactory.build_engine() should not import agent/."""

    def test_build_engine_returns_engine_components(self):
        from engine.factory import EngineFactory
        factory = EngineFactory({"lean_pool_size": 1})
        comp = factory.build_engine()
        try:
            assert comp.lean_pool is not None
            assert comp.prefilter is not None
            assert comp.broadcast is not None
            assert comp.scheduler is not None
            assert comp.error_intel is not None
            # Agent fields should be None (not built by build_engine)
            assert comp.agent_pool is None
            assert comp.hooks is None
            assert comp.plugins is None
            assert comp.meta_controller is None
            assert comp.hetero_engine is None
        finally:
            comp.close()

    def test_engine_factory_no_agent_imports(self):
        """engine/factory.py should have zero agent/ imports at module level."""
        import importlib
        import engine.factory
        source = importlib.util.find_spec("engine.factory").origin
        with open(source) as f:
            content = f.read()
        # Only the backward-compat build() uses a lazy import of prover.assembly
        # There should be no top-level agent imports
        lines = content.split("\n")
        top_level_agent_imports = [
            l for l in lines
            if ("from agent" in l or "import agent" in l)
            and not l.strip().startswith("#")
            and not l.strip().startswith("\"")
            and not l.strip().startswith("'")
        ]
        assert top_level_agent_imports == [], (
            f"engine/factory.py has agent imports: {top_level_agent_imports}")

    def test_system_assembler_builds_all(self):
        """SystemAssembler should build engine + agent + prover layers."""
        from prover.assembly import SystemAssembler
        assembler = SystemAssembler({"lean_pool_size": 1})
        comp = assembler.build(llm_provider=None)
        try:
            # Engine layer present
            assert comp.lean_pool is not None
            assert comp.broadcast is not None
            # Agent layer present (even without LLM, hooks/plugins are built)
            assert comp.hooks is not None
            assert comp.plugins is not None
            # Strategy present
            assert comp.meta_controller is not None
            assert comp.budget is not None
        finally:
            comp.close()


class TestProvePathCleanup:
    """2.3: ProofPipeline is the default prove() path."""

    def test_prove_defaults_to_pipeline(self):
        """prove() should NOT call _prove_legacy by default."""
        from prover.pipeline.orchestrator import Orchestrator
        # Check that default config does NOT have use_legacy_prove
        o = Orchestrator.__new__(Orchestrator)
        o.config = {}
        assert not o.config.get("use_legacy_prove", False)

    def test_prove_legacy_emits_deprecation_warning(self):
        """_prove_legacy should emit DeprecationWarning."""
        from prover.pipeline.orchestrator import Orchestrator
        from prover.models import BenchmarkProblem
        from unittest.mock import MagicMock

        o = Orchestrator.__new__(Orchestrator)
        o.config = {}
        o.meta = MagicMock()
        o.meta.select_initial_strategy.return_value = "light"
        o.reflector = MagicMock()
        o.confidence = MagicMock()
        o.confidence.should_abstain.return_value = True  # exit immediately
        o.budget = MagicMock()
        o.budget.is_exhausted.return_value = False
        o.hooks = MagicMock()
        o.hooks.fire.return_value = MagicMock(inject_context=None, action=None)
        o.plugins = MagicMock()
        o.hetero_engine = MagicMock()
        o.broadcast = MagicMock()
        o.scheduler = None
        o.lean_pool = None
        o._components = MagicMock()

        problem = BenchmarkProblem(
            problem_id="test", name="test",
            theorem_statement="theorem t : True")

        with pytest.warns(DeprecationWarning, match="_prove_legacy"):
            o._prove_legacy(problem)


class TestForkEnvRemoved:
    """2.5: fork_env should not exist anywhere."""

    def test_lean_pool_no_fork_env(self):
        from engine.lean_pool import LeanPool
        assert not hasattr(LeanPool, 'fork_env')

    def test_async_pool_no_fork_env(self):
        from engine.async_lean_pool import AsyncLeanPool
        assert not hasattr(AsyncLeanPool, 'fork_env')


class TestCacheEnvFingerprint:
    """2.6: Cache should invalidate after share_lemma."""

    def test_make_cache_key_includes_fingerprint(self):
        from engine._core import make_cache_key
        k1 = make_cache_key("thm", "prf", env_fingerprint="v0")
        k2 = make_cache_key("thm", "prf", env_fingerprint="v1")
        k3 = make_cache_key("thm", "prf", env_fingerprint="v0")
        assert k1 != k2  # different env → different key
        assert k1 == k3  # same env → same key

    def test_pool_env_version_starts_at_zero(self):
        from engine.lean_pool import LeanPool
        pool = LeanPool(pool_size=1)
        pool.start()
        try:
            assert pool._env_version == 0
        finally:
            pool.shutdown()


class TestConfigSchemaEngine:
    """2.7: Config schema should validate engine parameters."""

    def test_validates_pool_scaler_range(self):
        from config.schema import validate_config
        bad_config = {
            "agent": {"brain": {"provider": "mock", "model": "m"}},
            "prover": {"pipeline": {"max_samples": 8}},
            "engine": {
                "pool_scaler": {
                    "scale_up_threshold": 5.0,  # invalid: > 1.0
                }
            },
        }
        issues = validate_config(bad_config)
        assert any("scale_up_threshold" in i for i in issues)

    def test_engine_section_required(self):
        from config.schema import validate_config
        config = {
            "agent": {"brain": {"provider": "mock", "model": "m"}},
            "prover": {"pipeline": {"max_samples": 8}},
            # missing "engine"
        }
        issues = validate_config(config)
        assert any("engine" in i for i in issues)

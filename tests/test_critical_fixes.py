"""tests/test_critical_fixes.py — Tests for issues found in code review.

Covers:
  - _coerce_value edge cases (Fix #1)
  - Config validation raises on missing required sections (Fix #13)
  - AsyncCompileCache thread safety (Fix #2)
  - ElasticPool timeout behavior (Fix #8)
  - LocalTransport fallback state (Fix #5)
"""
from __future__ import annotations
import asyncio
import pytest


# ═══════════════════════════════════════════════════════════════
# Fix #1: _coerce_value
# ═══════════════════════════════════════════════════════════════

class TestCoerceValue:
    """Ensure environment variable type coercion is correct."""

    def _coerce(self, value):
        from config.schema import _coerce_value
        return _coerce_value(value)

    def test_zero_is_int_not_bool(self):
        assert self._coerce("0") == 0
        assert self._coerce("0") is not False

    def test_one_is_int_not_bool(self):
        assert self._coerce("1") == 1
        assert self._coerce("1") is not True

    def test_larger_ints(self):
        assert self._coerce("8") == 8
        assert self._coerce("-3") == -3
        assert self._coerce("1024") == 1024

    def test_floats(self):
        assert self._coerce("3.14") == 3.14
        assert self._coerce("0.0") == 0.0

    def test_booleans(self):
        assert self._coerce("true") is True
        assert self._coerce("True") is True
        assert self._coerce("yes") is True
        assert self._coerce("false") is False
        assert self._coerce("False") is False
        assert self._coerce("no") is False

    def test_none(self):
        assert self._coerce("null") is None
        assert self._coerce("none") is None
        assert self._coerce("") is None

    def test_strings(self):
        assert self._coerce("hello") == "hello"
        assert self._coerce("anthropic") == "anthropic"


# ═══════════════════════════════════════════════════════════════
# Fix #13: Config validation raises on missing required sections
# ═══════════════════════════════════════════════════════════════

class TestConfigValidation:

    def test_missing_required_section_raises(self):
        from config.schema import load_config
        # Empty config with no agent/prover/engine sections
        with pytest.raises(ValueError, match="Missing required"):
            load_config(path="/nonexistent/path.yaml", overrides={})

    def test_valid_config_no_raise(self):
        from config.schema import load_config
        valid = {
            "agent": {"brain": {"provider": "mock", "model": "mock"}},
            "prover": {"pipeline": {"max_samples": 10}},
            "engine": {},
        }
        # Should not raise
        cfg = load_config(path="/nonexistent/path.yaml", overrides=valid)
        assert cfg["agent"]["brain"]["provider"] == "mock"

    def test_range_violation_is_warning_not_error(self):
        from config.schema import load_config
        cfg_with_range_issue = {
            "agent": {"brain": {"provider": "mock", "model": "mock"}},
            "prover": {"pipeline": {"max_samples": 10, "temperature": 999.0}},
            "engine": {},
        }
        # Range violation should warn but not raise
        cfg = load_config(path="/nonexistent/path.yaml",
                          overrides=cfg_with_range_issue)
        assert cfg is not None


# ═══════════════════════════════════════════════════════════════
# Fix #2: AsyncCompileCache
# ═══════════════════════════════════════════════════════════════

class TestAsyncCompileCache:

    @pytest.mark.asyncio
    async def test_concurrent_first_access(self):
        """Multiple coroutines accessing cache for the first time
        should all use the same lock."""
        from engine._core import AsyncCompileCache, FullVerifyResult
        cache = AsyncCompileCache(maxsize=10)

        result = FullVerifyResult(success=True)

        async def writer(i):
            await cache.put(f"key_{i}", result)
            return await cache.get(f"key_{i}")

        # Run 20 concurrent writers — should not crash or lose data
        results = await asyncio.gather(*(writer(i) for i in range(20)))
        assert all(r is not None and r.success for r in results)
        assert cache.stats()["size"] == 10  # maxsize cap


# ═══════════════════════════════════════════════════════════════
# Fix #8: ElasticPool timeout
# ═══════════════════════════════════════════════════════════════

class TestElasticPoolTimeout:

    @pytest.mark.asyncio
    async def test_acquire_raises_on_timeout(self):
        """Pool should raise RuntimeError when all sessions are busy,
        not silently reuse a busy session."""
        from engine.remote_session import ElasticPool
        pool = ElasticPool(timeout_seconds=1)
        await pool.add_local(count=1)

        # Acquire the only session
        session = await pool._acquire()
        assert session.is_busy

        # Second acquire should timeout and raise
        with pytest.raises(RuntimeError, match="all .* sessions busy"):
            await pool._acquire()

        await pool._release(session)
        await pool.shutdown()


# ═══════════════════════════════════════════════════════════════
# Fix #5: LocalTransport fallback state
# ═══════════════════════════════════════════════════════════════

class TestLocalTransportFallback:

    @pytest.mark.asyncio
    async def test_no_repl_binary_sets_fallback_not_connected(self):
        """When no REPL binary exists, transport should be in fallback
        mode but is_connected should be False."""
        from engine.remote_session import LocalTransport
        transport = LocalTransport(
            project_dir="/nonexistent/path",
            timeout_seconds=5)
        ok = await transport.connect()
        assert ok is True  # graceful degradation
        assert transport._fallback is True
        assert transport.is_connected is False
        await transport.close()


# ═══════════════════════════════════════════════════════════════
# Round 2: SorryDetector block comments
# ═══════════════════════════════════════════════════════════════

class TestSorryDetectorBlockComments:

    def test_sorry_in_block_comment_is_ignored(self):
        from engine.prefilter import SorryDetector
        detector = SorryDetector()
        proof = """/- This is sorry but in a comment -/
intro x
simp"""
        result = detector.check(proof)
        assert result.passed, "sorry inside /- -/ block comment should be ignored"

    def test_sorry_outside_comment_is_detected(self):
        from engine.prefilter import SorryDetector
        detector = SorryDetector()
        proof = """intro x
sorry"""
        result = detector.check(proof)
        assert not result.passed

    def test_sorry_in_line_comment_is_ignored(self):
        from engine.prefilter import SorryDetector
        detector = SorryDetector()
        proof = """-- sorry is here as a comment
intro x
simp"""
        result = detector.check(proof)
        assert result.passed


# ═══════════════════════════════════════════════════════════════
# Round 2: _env_version_cache per project_dir
# ═══════════════════════════════════════════════════════════════

class TestEnvVersionCache:

    def test_different_project_dirs_get_different_cache(self):
        from prover.verifier.lean_repl import LeanREPL
        # Clear any stale cache
        LeanREPL._env_version_cache.clear()

        r1 = LeanREPL.__new__(LeanREPL)
        r1.project_dir = "/tmp/proj_a"
        r2 = LeanREPL.__new__(LeanREPL)
        r2.project_dir = "/tmp/proj_b"

        tag1 = r1._get_env_version_tag()
        tag2 = r2._get_env_version_tag()

        # Both should return a tag (even if no real files exist)
        assert isinstance(tag1, str) and len(tag1) > 0
        assert isinstance(tag2, str) and len(tag2) > 0

        # Cache should have separate entries
        assert "/tmp/proj_a" in LeanREPL._env_version_cache
        assert "/tmp/proj_b" in LeanREPL._env_version_cache


# ═══════════════════════════════════════════════════════════════
# Round 2: CAS bridge injection protection
# ═══════════════════════════════════════════════════════════════

class TestCASBridgeSanitization:

    def test_rejects_import_expression(self):
        from agent.tools.cas_bridge import CASBridge
        bridge = CASBridge()
        result = bridge._sage_eval("__import__('os').system('rm -rf /')", timeout=5)
        assert "rejected" in result.lower()

    def test_rejects_exec(self):
        from agent.tools.cas_bridge import CASBridge
        bridge = CASBridge()
        result = bridge._sage_eval("exec('malicious')", timeout=5)
        assert "rejected" in result.lower()

    def test_allows_normal_math(self):
        from agent.tools.cas_bridge import CASBridge
        bridge = CASBridge()
        # This will fail (no sage installed) but should NOT be rejected
        result = bridge._sage_eval("factorial(10)", timeout=5)
        assert "rejected" not in result.lower()


# ═══════════════════════════════════════════════════════════════
# Round 2: llm_provider threading import
# ═══════════════════════════════════════════════════════════════

class TestLLMProviderImports:

    def test_cached_provider_uses_proper_threading(self):
        """CachedProvider should use standard threading.Lock, not __import__ hack."""
        import inspect
        from agent.brain.llm_provider import CachedProvider
        source = inspect.getsource(CachedProvider)
        assert "__import__" not in source

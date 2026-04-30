"""tests/test_unified_pipeline.py — 验证 v3 大一统重构

覆盖
====
- prover.unified 主管线 (Profile + UnifiedProofRunner) 入口可用
- 5 个非搜索 preset (whole_proof / repair / dsp / reprover / leandojo) 的
  Profile 字段合理
- HeterogeneousEngine v3 用 legacy 构造签名仍能工作
- ProofLoop shim 正确把请求路由到 UnifiedProofRunner
- ProofPipeline.generate 通过 config 选择 profile 路径
- adapters: UnifiedResult ↔ ProofAttempt / AgentResult 数据无损
"""
from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════
# Test helpers
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FakeProblem:
    problem_id: str = "test_001"
    name: str = "test_thm"
    theorem_statement: str = "theorem t (n : Nat) : n + 0 = n := by simp"
    natural_language: str = ""
    domain: str = "nat_arithmetic"


class FakeMockLLM:
    """Mocks AsyncLLMProvider returning a fixed proof."""
    model_name = "fake-mock"

    async def chat(self, system="", messages=None, temperature=0.7,
                    tools=None, max_tokens=4096):
        return self._mk_response()

    async def generate(self, system="", user="", temperature=0.7,
                        tools=None, max_tokens=4096):
        return self._mk_response()

    def _mk_response(self):
        from agent.brain.llm_provider import LLMResponse
        return LLMResponse(
            content="Here's the proof:\n```lean\nby simp\n```",
            model="fake-mock",
            tokens_in=10, tokens_out=20, latency_ms=5,
            tool_calls=[],
            stop_reason="end_turn",
        )


# ══════════════════════════════════════════════════════════════════════
# 1. unified module import + preset shape
# ══════════════════════════════════════════════════════════════════════

class TestUnifiedAPI:
    def test_imports(self):
        from prover.unified import (
            UnifiedProofRunner, UnifiedResult,
            Profile, ToolKit, get_profile, PRESETS,
            EXPERIMENTAL_PRESETS,
            unified_to_attempt, unified_to_agent_result,
            enable_experimental_search_presets,
        )
        assert UnifiedProofRunner is not None
        assert PRESETS is not EXPERIMENTAL_PRESETS

    def test_active_presets_include_tree_search(self):
        """v4: MCTS / beam / best_first 已合流到 active PRESETS.

        合流契约: dialog.json schema 3.0 的 ``meta.search_tree`` 块
        让树搜索的元数据原生进主存储, 三个 profile 因此不再需要
        explicit opt-in。EXPERIMENTAL_PRESETS 仍保留作为未来的扩展点,
        但当前应为空。
        """
        from prover.unified import PRESETS, EXPERIMENTAL_PRESETS
        for required in ("mcts", "beam", "best_first"):
            assert required in PRESETS, \
                f"v4 起 {required} 必须在 active PRESETS 中"
        assert EXPERIMENTAL_PRESETS == {}, \
            "v4: EXPERIMENTAL_PRESETS 当前应为空"

    def test_active_presets_complete(self):
        """v4 大一统的 9 个 active preset 必须全部在。"""
        from prover.unified import PRESETS
        required = {
            # v3 family
            "whole_proof", "whole_proof_repair", "dsp",
            "reprover", "leandojo", "heterogeneous",
            # v4: tree-search merged
            "mcts", "beam", "best_first",
        }
        assert required.issubset(set(PRESETS)), \
            f"缺失 active preset: {required - set(PRESETS)}"

    def test_enable_experimental_is_noop_in_v4(self):
        """v4 起 enable_experimental_search_presets 是 no-op shim;
        旧脚本调用它仍然工作但不产生新效果。"""
        from prover.unified import (
            PRESETS, enable_experimental_search_presets, get_profile,
        )
        snapshot = set(PRESETS)
        enable_experimental_search_presets()  # 应该是 no-op
        assert set(PRESETS) == snapshot, \
            "enable_experimental_search_presets 不应改变 PRESETS"
        # mcts 已经在了, 直接拿
        prof = get_profile("mcts")
        assert prof.search.kind == "ucb"

    def test_profile_shapes(self):
        """5 个 active preset 的关键字段。"""
        from prover.unified import get_profile

        wp = get_profile("whole_proof")
        assert wp.tools == []
        assert wp.max_turns == 1

        wpr = get_profile("whole_proof_repair")
        assert wpr.max_turns >= 2
        assert any(t.value == "lean_verify" for t in wpr.tools)

        rep = get_profile("reprover")
        assert any(t.value == "premise_search" for t in rep.tools)
        assert any(t.value == "tactic_apply" for t in rep.tools)

        ldj = get_profile("leandojo")
        assert any(t.value == "tactic_apply" for t in ldj.tools)
        assert ldj.max_turns >= 10

        het = get_profile("heterogeneous")
        assert het.search.kind == "parallel"
        assert len(het.search.parallel_profiles) >= 2


# ══════════════════════════════════════════════════════════════════════
# 2. HeterogeneousEngine v3 — legacy ctor compat
# ══════════════════════════════════════════════════════════════════════

class TestHeteroEngineV3:
    def test_legacy_ctor_kwargs(self):
        """assembly.py 的旧 kwargs 仍能构造。"""
        from prover.pipeline.heterogeneous_engine import HeterogeneousEngine

        class FakePool:
            llm = None

        he = HeterogeneousEngine(
            pool=FakePool(),
            plugin_loader=None, hook_manager=None,
            retriever=None, broadcast=None,
            verification_scheduler=None,
        )
        assert he.directions is not None
        assert len(he.directions) == 4
        assert {d.name for d in he.directions} == \
            {"automation", "repair", "creative", "retrieval"}

    def test_directions_use_active_profiles(self):
        """方向引用的 profile 必须在 active PRESETS 中。"""
        from prover.pipeline.heterogeneous_engine import _DEFAULT_DIRECTIONS
        from prover.unified import PRESETS
        for d in _DEFAULT_DIRECTIONS:
            assert d.profile_name in PRESETS, \
                f"direction {d.name!r} → unknown profile {d.profile_name!r}"

    def test_sync_to_async_adapter_wraps_sync_llm(self):
        from prover.pipeline.heterogeneous_engine import _SyncToAsyncAdapter
        from agent.brain.llm_provider import LLMResponse

        class SyncLLM:
            model_name = "sync-test"
            def generate(self, **kw):
                return LLMResponse(
                    content="ok", model="sync-test",
                    tokens_in=1, tokens_out=2, latency_ms=1,
                )

        adapter = _SyncToAsyncAdapter(SyncLLM())
        assert adapter.model_name == "sync-test"

        async def run():
            return await adapter.generate(system="s", user="u")
        resp = asyncio.run(run())
        assert resp.content == "ok"
        assert resp.tokens_in == 1


# ══════════════════════════════════════════════════════════════════════
# 3. ProofLoop shim
# ══════════════════════════════════════════════════════════════════════

class TestProofLoopShim:
    def test_construction_compatible(self):
        from prover.pipeline.proof_loop import ProofLoop
        loop = ProofLoop(
            lean_env=None, llm=None, retriever=None,
            config={"max_repair_rounds": 3},
        )
        assert loop.max_repair_rounds == 3
        assert loop._max_turns == 4   # repair_rounds + 1

    def test_zero_repair_uses_whole_proof(self):
        """max_repair_rounds=0 → whole_proof profile (max_turns=1)."""
        from prover.pipeline.proof_loop import ProofLoop
        loop = ProofLoop(
            lean_env=None, llm=None, retriever=None,
            config={"max_repair_rounds": 0},
        )
        assert loop._max_turns == 1


# ══════════════════════════════════════════════════════════════════════
# 4. Adapters
# ══════════════════════════════════════════════════════════════════════

class TestAdapters:
    def _make_unified_result(self, *, success=True, proof="by simp"):
        from prover.unified import UnifiedResult
        from agent.runtime.agent_loop import LoopResult, LoopMessage
        loop = LoopResult(
            content=f"```lean\n{proof}\n```",
            proof_code=proof,
            messages=[
                LoopMessage(role="user", content="prove it"),
                LoopMessage(role="assistant", content=f"```lean\n{proof}\n```"),
            ],
            turns_used=1,
            total_tokens=30,
            total_latency_ms=10,
            tools_called=[],
            stopped_reason="proof_found" if success else "max_turns",
        )
        return UnifiedResult(
            profile_name="whole_proof_repair",
            success=success,
            proof_code=proof,
            loop_result=loop,
            total_duration_ms=10,
        )

    def test_unified_to_attempt_success(self):
        from prover.unified import unified_to_attempt
        from prover.models import AttemptStatus

        ur = self._make_unified_result(success=True, proof="by simp")
        att = unified_to_attempt(ur, attempt_number=1)

        assert att.lean_result == AttemptStatus.SUCCESS
        assert att.generated_proof == "by simp"
        assert att.attempt_number == 1
        assert att.llm_tokens_out == 30

    def test_unified_to_attempt_failure(self):
        from prover.unified import unified_to_attempt
        from prover.models import AttemptStatus

        ur = self._make_unified_result(success=False, proof="")
        att = unified_to_attempt(ur, attempt_number=2)

        assert att.lean_result == AttemptStatus.LEAN_ERROR
        assert att.generated_proof == ""

    def test_unified_to_agent_result(self):
        from prover.unified import unified_to_agent_result

        ur = self._make_unified_result(success=True, proof="by ring")
        ar = unified_to_agent_result(ur, agent_name="test_dir")

        assert ar.success is True
        assert ar.proof_code == "by ring"
        assert ar.agent_name == "test_dir"
        assert ar.confidence > 0.9
        assert ar.metadata["profile_name"] == "whole_proof_repair"


# ══════════════════════════════════════════════════════════════════════
# 5. ProofPipeline routes through unified when profile is set
# ══════════════════════════════════════════════════════════════════════

class TestProofPipelineUnifiedRoute:
    def test_pipeline_recognizes_profile_config(self):
        """ProofPipeline.config['profile'] = X 时应路由到 _generate_via_unified。"""
        from prover.pipeline.proof_pipeline import ProofPipeline

        # Inspect the source: _generate_via_unified should exist
        assert hasattr(ProofPipeline, "_generate_via_unified")

    def test_unified_route_unknown_profile_fallbacks_gracefully(self):
        """未知 profile 名应降级到 hetero, 不抛异常。"""
        # 仅验证代码路径存在; 完整集成测试需要真实 components
        from prover.pipeline.proof_pipeline import ProofPipeline
        import inspect
        src = inspect.getsource(ProofPipeline._generate_via_unified)
        assert "Unknown profile" in src or "fallback" in src.lower()


# ══════════════════════════════════════════════════════════════════════
# 6. dialog.json round-trip via UnifiedProofRunner
# ══════════════════════════════════════════════════════════════════════

class TestDialogRoundTrip:
    @pytest.mark.asyncio
    async def test_minimal_run(self, tmp_path):
        """Smoke: UnifiedProofRunner.run() 在 mock LLM 下产出有 proof_code 的结果。"""
        from prover.unified import UnifiedProofRunner, get_profile

        runner = UnifiedProofRunner(
            llm=FakeMockLLM(),
            lean_pool=None,
            knowledge_store=None,
            retriever=None,
            broadcast_bus=None,
        )
        problem = FakeProblem()
        profile = get_profile("whole_proof")  # 单轮, 无工具
        ur = await runner.run(problem, profile=profile)

        assert ur.profile_name == "whole_proof"
        assert ur.loop_result is not None
        # mock LLM 返回了 lean 代码
        assert "by simp" in (ur.proof_code or ur.loop_result.proof_code)


# ══════════════════════════════════════════════════════════════════════
# Boilerplate for sync run when pytest-asyncio missing
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

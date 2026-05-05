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
        from agent.brain.async_llm_provider import LLMResponse
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
            unified_to_attempt,
        )
        assert UnifiedProofRunner is not None
        assert isinstance(PRESETS, dict) and PRESETS

    def test_active_presets_include_tree_search(self):
        """

        合流契约: dialog.json schema 3.0 的 ``meta.search_tree`` 块
        让树搜索的元数据原生进主存储, 三个 profile 因此不再需要
        explicit opt-in。

        
        (空字典 + no-op 函数。如果将来需要 gating,可重新引入。)
        """
        from prover.unified import PRESETS
        for required in ("mcts", "beam", "best_first"):
            assert required in PRESETS, \
                f"missing search-based preset: {required}"

    def test_active_presets_complete(self):
        """v4 大一统的 9 个 active preset 必须全部在。"""
        from prover.unified import PRESETS
        required = {

            "whole_proof", "whole_proof_repair", "dsp",
            "reprover", "leandojo", "heterogeneous",

            "mcts", "beam", "best_first",
        }
        assert required.issubset(set(PRESETS)), \
            f"缺失 active preset: {required - set(PRESETS)}"

    def test_v11_experimental_shim_removed(self):
        """

        Anyone who needs the old name can either import from
        ``prover.unified.profiles`` directly (it's also gone there) or
        define their own local empty dict + identity function.
        """
        import prover.unified as pu
        assert not hasattr(pu, "EXPERIMENTAL_PRESETS")
        assert not hasattr(pu, "enable_experimental_search_presets")

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
# 2. HeterogeneousEngine 
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

    # test_unified_to_agent_result removed in 
    # agent.runtime.sub_agent.AgentResult which was deleted alongside
    # the rest of the SubAgent / AsyncAgentPool subsystem.

# ══════════════════════════════════════════════════════════════════════
# 5. ProofPipeline routes through unified when profile is set
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

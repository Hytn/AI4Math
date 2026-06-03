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
from types import SimpleNamespace

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


class TestAgentLoopToolCallOrdering:
    @pytest.mark.asyncio
    async def test_tool_calls_are_answered_before_proof_stop(self):
        from agent.brain.async_llm_provider import LLMResponse
        from agent.persistence.dialog_format import validate_dialog
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.tools.base import Tool, ToolContext, ToolResult
        from agent.tools.registry import ToolRegistry

        class OneToolCallLLM:
            model_name = "tool-call-llm"

            async def generate(self, system="", user="", temperature=0.7,
                               tools=None, max_tokens=4096):
                return await self.chat(
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    tools=tools,
                    max_tokens=max_tokens,
                )

            async def chat(self, system="", messages=None, temperature=0.7,
                           tools=None, max_tokens=4096):
                return LLMResponse(
                    content="I found a proof.\n```lean\nby simp\n```",
                    model=self.model_name,
                    tokens_in=1,
                    tokens_out=1,
                    latency_ms=1,
                    tool_calls=[{
                        "id": "call_1",
                        "name": "premise_search",
                        "input": {"query": "simp"},
                    }],
                    stop_reason="tool_use",
                )

        class PremiseSearchTool(Tool):
            name = "premise_search"
            description = "Search premises."
            input_schema = {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }

            async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
                return ToolResult.success("premise result")

        registry = ToolRegistry()
        registry.register(PremiseSearchTool())
        loop = AgentLoop(
            llm=OneToolCallLLM(),
            tools=registry,
            config=LoopConfig(max_turns=1, stop_on_proof=True),
        )

        result = await loop.run(system_prompt="", initial_message="prove it")
        dialog = result.to_dialog(problem_id="tool_order")

        assert result.stopped_reason == "max_turns"
        assert result.tools_called == ["premise_search"]
        assert [m.role for m in result.messages] == [
            "user", "assistant", "tool_result"]
        assert validate_dialog(dialog) == []

    @pytest.mark.asyncio
    async def test_lean_verify_success_stops_on_last_turn(self):
        import json
        from agent.brain.async_llm_provider import LLMResponse
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.tools.base import Tool, ToolContext, ToolResult
        from agent.tools.registry import ToolRegistry

        proof = "theorem t : True := by trivial"

        class OneLeanVerifyLLM:
            model_name = "tool-call-llm"

            async def chat(self, system="", messages=None, temperature=0.7,
                           tools=None, max_tokens=4096):
                return LLMResponse(
                    content="checking",
                    model=self.model_name,
                    tokens_in=1,
                    tokens_out=1,
                    latency_ms=1,
                    tool_calls=[{
                        "id": "call_1",
                        "name": "lean_verify",
                        "input": {"code": proof},
                    }],
                    stop_reason="tool_use",
                )

        class LeanVerifyTool(Tool):
            name = "lean_verify"
            description = "Verify Lean."
            input_schema = {"type": "object", "properties": {}}

            async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
                return ToolResult.success(json.dumps({
                    "verified": True,
                    "sorry_free": True,
                    "errors": [],
                    "goals_remaining": [],
                }))

        registry = ToolRegistry()
        registry.register(LeanVerifyTool())
        loop = AgentLoop(
            llm=OneLeanVerifyLLM(),
            tools=registry,
            config=LoopConfig(max_turns=1, stop_on_proof=True),
        )

        result = await loop.run(system_prompt="", initial_message="prove it")

        assert result.stopped_reason == "proof_found"
        assert result.proof_code == proof
        assert result.to_dialog(problem_id="p")["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_lean_verify_integrity_violation_does_not_stop(self):
        import json
        from agent.brain.async_llm_provider import LLMResponse
        from agent.runtime.agent_loop import AgentLoop, LoopConfig
        from agent.tools.base import Tool, ToolContext, ToolResult
        from agent.tools.registry import ToolRegistry

        class OneLeanVerifyLLM:
            model_name = "tool-call-llm"

            async def chat(self, system="", messages=None, temperature=0.7,
                           tools=None, max_tokens=4096):
                return LLMResponse(
                    content="checking",
                    model=self.model_name,
                    tokens_in=1,
                    tokens_out=1,
                    latency_ms=1,
                    tool_calls=[{
                        "id": "call_1",
                        "name": "lean_verify",
                        "input": {"code": "theorem t : True := by native_decide"},
                    }],
                    stop_reason="tool_use",
                )

        class LeanVerifyTool(Tool):
            name = "lean_verify"
            description = "Verify Lean."
            input_schema = {"type": "object", "properties": {}}

            async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
                return ToolResult.success(json.dumps({
                    "verified": True,
                    "sorry_free": True,
                    "errors": [],
                    "integrity_violations": ["[critical] Uses native_decide"],
                }))

        registry = ToolRegistry()
        registry.register(LeanVerifyTool())
        loop = AgentLoop(
            llm=OneLeanVerifyLLM(),
            tools=registry,
            config=LoopConfig(max_turns=1, stop_on_proof=True),
        )

        result = await loop.run(system_prompt="", initial_message="prove it")

        assert result.stopped_reason == "max_turns"
        assert result.to_dialog(problem_id="p")["result"]["success"] is False


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
        assert ldj.observation.auto_inject_lean_compile is False

        het = get_profile("heterogeneous")
        assert het.search.kind == "parallel"
        assert len(het.search.parallel_profiles) >= 2

    def test_leandojo_initial_message_is_step_level_only(self):
        from prover.unified import UnifiedProofRunner, get_profile

        runner = UnifiedProofRunner(llm=FakeMockLLM())
        msg = runner._build_initial_message(FakeProblem(), get_profile("leandojo"))
        assert "Call `tactic_apply`" in msg
        assert "Do NOT output a full proof" in msg
        assert "Output the final proof" not in msg

    def test_step_level_rejects_single_shot_pool(self):
        from prover.unified import UnifiedProofRunner

        class SingleShotPool:
            def stats(self):
                return {
                    "active_sessions": 4,
                    "all_fallback": False,
                    "all_single_shot": True,
                }

        runner = UnifiedProofRunner(llm=FakeMockLLM(), lean_pool=SingleShotPool())
        assert runner._lean_pool_supports_tactics() is False

    @pytest.mark.asyncio
    async def test_step_level_bootstrap_passes_problem_preamble(self):
        from agent.tools.base import ToolContext
        from prover.unified import UnifiedProofRunner

        class PreamblePool:
            def __init__(self):
                self.calls = []

            def stats(self):
                return {
                    "active_sessions": 1,
                    "all_fallback": False,
                    "all_single_shot": False,
                }

            async def start_proof(self, theorem: str, preamble: str = ""):
                self.calls.append((theorem, preamble))
                return SimpleNamespace(
                    success=True, new_env_id=7, remaining_goals=["goal"])

        pool = PreamblePool()
        runner = UnifiedProofRunner(llm=FakeMockLLM(), lean_pool=pool)
        problem = SimpleNamespace(
            theorem_statement="theorem t : True",
            lean_preamble="import Mathlib\nopen Nat",
        )
        ctx = ToolContext()

        boot = await runner._bootstrap_step_level_state(problem, ctx)

        assert boot is None
        assert pool.calls == [("theorem t : True", "import Mathlib\nopen Nat")]
        assert ctx.shared_state["proof_state_id"] == 7
        assert ctx.current_goals == ["goal"]

    def test_parse_lean_files_preserves_file_preamble(self, tmp_path):
        from benchmarks.datasets._base import parse_lean_files

        lean_file = tmp_path / "T.lean"
        lean_file.write_text(
            "import Mathlib\n\n"
            "set_option maxHeartbeats 0\n\n"
            "open Nat\n\n"
            "theorem t (n : Nat) : n = n := by rfl\n",
            encoding="utf-8",
        )

        problems = parse_lean_files(
            [lean_file], problem_id_prefix="p_", source="unit")

        assert len(problems) == 1
        assert problems[0].lean_preamble == (
            "import Mathlib\n\n"
            "set_option maxHeartbeats 0\n\n"
            "open Nat")

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
# 4b. whole_proof auto verification
# ══════════════════════════════════════════════════════════════════════

class CapturingLeanPool:
    def __init__(self):
        self.calls = []

    async def verify_complete(self, theorem: str, proof: str,
                              preamble: str = ""):
        self.calls.append((theorem, proof, preamble))
        return SimpleNamespace(success=True, has_sorry=False, errors=[])


class TestAutoVerifyProof:
    @pytest.mark.asyncio
    async def test_splits_full_theorem_block_before_verify_complete(self):
        from prover.unified import UnifiedProofRunner

        pool = CapturingLeanPool()
        runner = UnifiedProofRunner(llm=None, lean_pool=pool)
        problem = FakeProblem(theorem_statement="theorem t : True")

        verified = await runner._auto_verify_proof(
            problem, "theorem t : True := by\n  trivial")

        assert verified is True
        theorem, proof, preamble = pool.calls[-1]
        assert theorem == "theorem t : True"
        assert proof.strip() == ":= by\n  trivial"
        assert preamble == ""

    @pytest.mark.asyncio
    async def test_keeps_problem_statement_for_proof_body(self):
        from prover.unified import UnifiedProofRunner

        pool = CapturingLeanPool()
        runner = UnifiedProofRunner(llm=None, lean_pool=pool)
        problem = FakeProblem(theorem_statement="theorem t : True")

        verified = await runner._auto_verify_proof(problem, "by\n  trivial")

        assert verified is True
        assert pool.calls[-1] == ("theorem t : True", "by\n  trivial", "")


class TestRunnerAutoVerifyEvidence:
    @pytest.mark.asyncio
    async def test_runner_records_auto_verify_evidence(self):
        from prover.unified import UnifiedProofRunner, get_profile

        pool = CapturingLeanPool()
        runner = UnifiedProofRunner(llm=FakeMockLLM(), lean_pool=pool)
        problem = FakeProblem(theorem_statement="theorem t : True")

        ur = await runner.run(problem, profile=get_profile("whole_proof"))

        assert ur.success is True
        assert ur.loop_result.auto_verify["verified"] is True
        dialog = ur.loop_result.to_dialog(problem_id="p")
        assert dialog["result"]["extra"]["auto_verify"]["verified"] is True


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

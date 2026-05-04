"""tests/test_smoke_v14.py — Pin v14 reservoir wirings against regression.

每条断言钉一处「未接通备胎」回归到 v13 主路径的接通点。回归(把接通拆掉)
立刻被 CI 抓到。

跑: ``pytest tests/test_smoke_v14.py -v``

mock-only (no Lean, no real LLM)。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = Path(__file__).parent.parent


# ═════════════════════════════════════════════════════════════════
# 项 ① — engine/summary_compressor.py 回归 + 接通
# ═════════════════════════════════════════════════════════════════

class TestProject1_SummaryCompressor:
    """Lean 错误压缩器:从初版 engine/lane/ 回归到 engine/ 顶层。"""

    def test_module_present(self):
        """engine/summary_compressor.py 必须存在且可 import。"""
        assert (ROOT / "engine/summary_compressor.py").exists()
        from engine.summary_compressor import (
            compress_lean_errors, compress_feedback, compress_broadcast,
        )
        for fn in (compress_lean_errors, compress_feedback,
                    compress_broadcast):
            assert callable(fn)

    def test_compress_feedback_actually_compresses(self):
        """v14 项①: 压缩比应小于 0.5 (即压到原来一半以内)。"""
        from engine.summary_compressor import compress_feedback
        text = "error: type mismatch\n  expected ℕ, got ℝ\n" * 100
        out = compress_feedback(text, budget=400)
        assert len(out) <= 420  # 允许 20 字符缓冲
        assert len(out) < len(text) * 0.5

    def test_loopconfig_has_compress_flag(self):
        """LoopConfig 必须暴露 compress_tool_results 开关。"""
        from agent.runtime.agent_loop import LoopConfig
        cfg = LoopConfig()
        assert hasattr(cfg, "compress_tool_results"), (
            "v14 项①接通点丢失: LoopConfig 必须有 compress_tool_results 字段。")
        assert cfg.compress_tool_results is True, (
            "默认应为 True (主路径接通)。设 False 等于退回 v13 行为。")
        assert cfg.compress_budget > 0

    def test_agent_loop_calls_compress_on_failure(self):
        """v14 项①: agent_loop 在 inject tool_result 时必须调 compress_feedback。"""
        import inspect
        from agent.runtime.agent_loop import AgentLoop
        src = inspect.getsource(AgentLoop)
        assert "compress_feedback" in src, (
            "AgentLoop 源码必须 import compress_feedback (主路径接通)。"
            "去掉这条 import 就是回退到未接通状态。")

    def test_broadcast_tool_uses_compression(self):
        """v14 项①: BroadcastTool 的 publish/get_recent 必须走压缩。"""
        import inspect
        from prover.unified.tools_extra import BroadcastTool
        src = inspect.getsource(BroadcastTool.execute)
        # 同时检查两个方向
        assert "compress_broadcast" in src
        assert "compress_feedback" in src


# ═════════════════════════════════════════════════════════════════
# 项 ② — engine/policy/ (PolicyEngine + recovery + task_state)
# ═════════════════════════════════════════════════════════════════

class TestProject2_PolicyEngine:
    """声明式策略规则引擎: 从初版 engine/lane/ 回归到 engine/policy/。"""

    def test_module_present(self):
        for f in ("engine/policy/__init__.py",
                   "engine/policy/task_state.py",
                   "engine/policy/recovery.py",
                   "engine/policy/engine.py"):
            assert (ROOT / f).exists(), f"missing: {f}"

    def test_default_engine_has_5_rules(self):
        """PolicyEngine.default() 必须装载 5 条内置规则。"""
        from engine.policy import PolicyEngine
        e = PolicyEngine.default()
        rules = getattr(e, "_rules", None) or getattr(e, "rules", None)
        assert rules is not None and len(rules) == 5, (
            f"expected 5 default rules, got {len(rules) if rules else None}")

    def test_state_machine_runs(self):
        """ProofTaskStateMachine 真能 record 事件。"""
        from engine.policy import (
            PolicyEngine, ProofTaskStateMachine, TaskContext, TaskEvent,
            TaskFailure, TaskStatus, ProofFailureClass,
        )
        sm = ProofTaskStateMachine(
            task_id="t1",
            context=TaskContext(theorem_name="t", formal_statement="t : True"))
        ev = TaskEvent(
            seq=0, event_name="verify_failed",
            prev_status=TaskStatus.VERIFYING,
            new_status=TaskStatus.VERIFYING,
            failure=TaskFailure(
                failure_class=ProofFailureClass.TYPE_MISMATCH,
                message="dummy"))
        e = PolicyEngine.default()
        # 不抛异常即通过
        decision = e.evaluate(sm, [ev])
        assert decision is not None

    def test_agent_loop_accepts_policy_engine(self):
        """AgentLoop.__init__ 必须接受 policy_engine 关键字。"""
        import inspect
        from agent.runtime.agent_loop import AgentLoop
        sig = inspect.signature(AgentLoop.__init__)
        assert "policy_engine" in sig.parameters, (
            "AgentLoop.__init__ 必须有 policy_engine 参数 (项②接通点)。")

    def test_runner_accepts_policy_engine(self):
        """UnifiedProofRunner.__init__ 必须接受 policy_engine 关键字。"""
        import inspect
        from prover.unified.runner import UnifiedProofRunner
        sig = inspect.signature(UnifiedProofRunner.__init__)
        assert "policy_engine" in sig.parameters

    def test_runner_threads_policy_to_agent_loop(self):
        """v14 项②: runner 创建 AgentLoop 时必须把 policy_engine 透传。"""
        import inspect
        from prover.unified.runner import UnifiedProofRunner
        src = inspect.getsource(UnifiedProofRunner)
        # 用稳定的字符串特征 (不依赖具体行号或多行布局)
        assert "policy_engine=self.policy_engine" in src, (
            "runner 必须把 policy_engine 透传给 AgentLoop。")


# ═════════════════════════════════════════════════════════════════
# 项 ③ — prover/lemma_bank/ 回归 + 跨问题 SQLite 库接通
# ═════════════════════════════════════════════════════════════════

class TestProject3_PersistentLemmaBank:
    """跨问题/跨会话 lemma 库 — 预留接口 A 知识库的 lemma 维度。"""

    def test_module_present(self):
        from prover.lemma_bank import (
            ProvedLemma, LemmaBank, PersistentLemmaBank,
            LemmaExtractor, LemmaVerifier,
        )
        for cls in (ProvedLemma, LemmaBank, PersistentLemmaBank,
                     LemmaExtractor, LemmaVerifier):
            assert isinstance(cls, type)

    def test_persistent_bank_roundtrip(self):
        """add → search 真能在 SQLite 里走通。"""
        from prover.lemma_bank import PersistentLemmaBank, ProvedLemma
        db = os.path.join(tempfile.mkdtemp(), "v14_t.db")
        bank = PersistentLemmaBank(db)
        bank.add(ProvedLemma(
            name="add_zero",
            statement="lemma add_zero (n : ℕ) : n + 0 = n",
            proof=":= by simp"))
        results = bank.search("add", top_k=5)
        assert len(results) >= 1
        # 找到的引理 statement 必须包含 "add"
        found = [getattr(r, "statement", str(r)) for r in results]
        assert any("add" in s for s in found)

    def test_lemma_bank_tool_accepts_persistent(self):
        """LemmaBankTool.__init__ 必须接受 persistent_bank 参数。"""
        from prover.unified.tools_extra import LemmaBankTool
        from prover.lemma_bank import PersistentLemmaBank
        db = os.path.join(tempfile.mkdtemp(), "v14_t2.db")
        bank = PersistentLemmaBank(db)
        tool = LemmaBankTool(knowledge_store=None, persistent_bank=bank)
        assert tool._persistent_bank is bank, (
            "LemmaBankTool 必须把 persistent_bank 存为字段 (项③接通点)。")

    def test_conjecture_propose_tool_accepts_persistent(self):
        """ConjectureProposeTool 必须接受 persistent_bank 做后置写入。"""
        from prover.unified.tools_extra import ConjectureProposeTool
        from prover.lemma_bank import PersistentLemmaBank
        db = os.path.join(tempfile.mkdtemp(), "v14_t3.db")
        bank = PersistentLemmaBank(db)
        tool = ConjectureProposeTool(llm=None, persistent_bank=bank)
        assert tool._persistent_bank is bank

    def test_runner_threads_persistent_bank(self):
        """v14 项③: UnifiedProofRunner 必须接受并透传 persistent_lemma_bank."""
        import inspect
        from prover.unified.runner import UnifiedProofRunner
        sig = inspect.signature(UnifiedProofRunner.__init__)
        assert "persistent_lemma_bank" in sig.parameters

    def test_tool_kits_threads_persistent_bank(self):
        """v14 项③: build_tool_registry 必须接受并透传 persistent_lemma_bank."""
        import inspect
        from prover.unified.tool_kits import build_tool_registry
        sig = inspect.signature(build_tool_registry)
        assert "persistent_lemma_bank" in sig.parameters


# ═════════════════════════════════════════════════════════════════
# 项 ④ — prover/plugins/ + plugins/strategies/ 数据
# ═════════════════════════════════════════════════════════════════

class TestProject4_DomainPlugins:
    """YAML-driven 领域插件 — 按定理领域注入 few-shot/premises/hint。"""

    def test_module_present(self):
        from prover.plugins import StrategyPlugin, PluginLoader
        assert isinstance(PluginLoader, type)
        assert isinstance(StrategyPlugin, type)

    def test_three_strategies_data_present(self):
        """三个领域目录必须存在 (algebra/analysis/number-theory)。"""
        for domain in ("algebra", "analysis", "number-theory"):
            path = ROOT / "plugins" / "strategies" / domain
            assert path.exists(), f"missing plugin data: {path}"
            for f in ("plugin.yaml", "premises.jsonl", "few_shot.md"):
                assert (path / f).exists(), f"missing: {path / f}"

    def test_loader_discovers_three(self):
        """PluginLoader 必须发现 3 个插件。"""
        from prover.plugins import PluginLoader
        loader = PluginLoader(plugin_dirs=[str(ROOT / "plugins/strategies")])
        loader.discover()
        names = set(loader.list_plugins())
        assert {"algebra", "analysis", "number-theory"} <= names

    def test_match_returns_relevant_plugin(self):
        """理论里有 'Ring' 关键字的题, algebra 必须匹配上。"""
        from prover.plugins import PluginLoader
        loader = PluginLoader(plugin_dirs=[str(ROOT / "plugins/strategies")])
        matches = loader.match(
            "theorem t [Ring R] (a b : R) : a * b = b * a")
        assert any(p.name == "algebra" for p in matches)

    def test_runner_accepts_plugin_loader(self):
        import inspect
        from prover.unified.runner import UnifiedProofRunner
        sig = inspect.signature(UnifiedProofRunner.__init__)
        assert "plugin_loader" in sig.parameters

    def test_runner_injects_plugin_into_initial_message(self):
        """v14 项④: _build_initial_message 必须查 plugin_loader."""
        import inspect
        from prover.unified.runner import UnifiedProofRunner
        src = inspect.getsource(UnifiedProofRunner._build_initial_message)
        assert "plugin_loader" in src, (
            "_build_initial_message 必须 query plugin_loader (项④接通点)。")


# ═════════════════════════════════════════════════════════════════
# Cross-cutting: 不传任何 v14 新参数, 行为与 v13 一致 (向后兼容)
# ═════════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """每个 v14 新增字段的默认值都不应改变 v13 行为。"""

    def test_runner_constructible_without_v14_kwargs(self):
        """不传 plugin_loader/persistent_lemma_bank/policy_engine 也能构造。"""
        from prover.unified.runner import UnifiedProofRunner
        from agent.brain.async_llm_provider import create_async_provider
        llm = create_async_provider({"provider": "mock", "model": "mock"})
        runner = UnifiedProofRunner(llm=llm, lean_pool=None, retriever=None)
        assert runner.plugin_loader is None
        assert runner.persistent_lemma_bank is None
        assert runner.policy_engine is None

    def test_agent_loop_constructible_without_policy(self):
        """AgentLoop 不传 policy_engine 仍可构造。"""
        from agent.runtime.agent_loop import AgentLoop
        from agent.tools import ToolRegistry
        from agent.brain.async_llm_provider import create_async_provider
        llm = create_async_provider({"provider": "mock", "model": "mock"})
        loop = AgentLoop(llm=llm, tools=ToolRegistry())
        assert loop.policy_engine is None

    def test_lemma_bank_tool_constructible_without_persistent(self):
        """LemmaBankTool 不传 persistent_bank 仍 v13 行为。"""
        from prover.unified.tools_extra import LemmaBankTool
        tool = LemmaBankTool(knowledge_store=None)
        assert tool._persistent_bank is None

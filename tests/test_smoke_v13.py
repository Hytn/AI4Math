"""tests/test_smoke_v13.py — Pin v13 cleanup + bug fixes against regression.

每条断言对应 v13 修的一个 bug 或删的一个死代码块。回归(把代码改回去)
立刻被 CI 抓到。

跑: ``pytest tests/test_smoke_v13.py -v``

mock-only (no Lean, no real LLM)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────
# A. Latent bug fixes
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_goal_decomposer_is_async():
    """v13 fix: GoalDecomposer.decompose 必须是 async coroutine 函数.

    v12 之前 decompose 是 sync 但内部调 ``self.llm.generate(...)`` 这条
    AsyncLLMProvider 上是 async 接口。``resp.content`` 必触
    AttributeError, dsp / pantograph_dsp / conjecture_driven 三个 profile
    在 anthropic provider 下永远跑不通。回归一旦把 async 关键字去掉,
    这条断言会立即失败。
    """
    import inspect
    from prover.decompose.goal_decomposer import GoalDecomposer
    assert inspect.iscoroutinefunction(GoalDecomposer.decompose), (
        "GoalDecomposer.decompose must be `async def` to handle "
        "AsyncLLMProvider.generate (which returns a coroutine). "
        "Reverting to sync reintroduces the v12 latent bug.")


@pytest.mark.asyncio
async def test_decompose_subgoal_tool_awaits_decomposer():
    """v13 fix: DecomposeSubgoalTool 必须 await decomposer.decompose.

    这是 GoalDecomposer 修复的另一半 —— 调用方也得 await。
    """
    import inspect as _inspect
    src = _inspect.getsource(
        __import__("prover.unified.tools_extra",
                    fromlist=["DecomposeSubgoalTool"])
    )
    # 找 execute body 里的 decompose 调用, 必须紧跟 await
    assert "await decomposer.decompose(" in src, (
        "DecomposeSubgoalTool.execute must `await decomposer.decompose(...)`; "
        "missing await reintroduces the sync-call-async latent bug.")


@pytest.mark.asyncio
async def test_heterogeneous_propagates_broadcast_tool():
    """v13 fix: heterogeneous _run_parallel 必须把 BROADCAST 注入 sub-profile.

    v12 之前 sub-profile 用 ``sp.__dict__`` 实例化, parent profile 的
    ``ToolKit.BROADCAST`` 不传播 —— 整个 ``engine/broadcast.py`` 在主路径
    死代码, heterogeneous 实际只是 best-of-4。
    """
    import inspect as _inspect
    src = _inspect.getsource(
        __import__("prover.unified.runner", fromlist=["UnifiedProofRunner"])
    )
    # 找 _run_parallel 里把 BROADCAST 加到 tools 的代码段
    assert "ToolKit.BROADCAST" in src and "_augment" in src, (
        "_run_parallel must augment each sub-profile's tools list with "
        "ToolKit.BROADCAST. Without this, the broadcast bus is unreachable "
        "from sub-runners and heterogeneous degenerates to best-of-N."
    )


# ─────────────────────────────────────────────────────────────────
# B. Dead code deletion (each assert pins a removal)
# ─────────────────────────────────────────────────────────────────

class TestDeadCodeRemoved:
    """每条对应 v13 删除的一个文件 / 模块。回归即重新引入死代码。"""

    def test_index_html_is_github_pages_landing(self):
        """index.html 是 GitHub Pages 主页, 不是死代码 (v13 一度误删, 已恢复).

        如果未来真要删, 必须先迁移 GitHub Pages source。这条断言钉住主页
        存在 + 反映 v13 的内容, 防止下一轮"死代码大扫除"再次误伤。
        """
        idx = ROOT / "index.html"
        assert idx.exists(), (
            "index.html is the GitHub Pages landing for this repo. "
            "Do not delete without first migrating Pages source.")
        text = idx.read_text(encoding="utf-8")
        # 简单的内容对齐检查: 反映 v13 的核心定位
        assert "AI4Math" in text
        assert "三件核心" in text or "Reserved Interfaces" in text or "预留接口" in text, (
            "index.html should reflect v13's 3-core + 3-reserved-interface "
            "framing. If you re-themed the page, update this assertion.")

    def test_redundant_arch_html_gone(self):
        assert not (ROOT / "docs/architecture.html").exists()
        assert not (ROOT / "docs/architecture.svg").exists()

    def test_old_cleanup_docs_merged(self):
        """三份历次 CLEANUP_*.md 合并进 CHANGELOG。"""
        for name in ("CLEANUP_v9.md", "CLEANUP_v10.md", "CLEANUP_SUMMARY.md"):
            assert not (ROOT / "docs" / name).exists(), (
                f"docs/{name} should have been merged into CHANGELOG.md")

    def test_lean3_to_lean4_module_removed(self):
        """0 主路径调用方的 Lean3→Lean4 重命名表已删。"""
        assert not (ROOT / "engine/lean3_to_lean4.py").exists()

    def test_repl_protocol_module_removed(self):
        """0 主路径 import 的 REPL wire-format types 已删。"""
        assert not (ROOT / "engine/repl_protocol.py").exists()

    def test_prompt_builder_module_removed(self):
        """整个 prompt_builder.py 109 行死代码删除; FEW_SHOT_EXAMPLES 挪到 few_shot.py."""
        assert not (ROOT / "common/prompt_builder.py").exists()
        from common.few_shot import FEW_SHOT_EXAMPLES
        assert "induction" in FEW_SHOT_EXAMPLES.lower()

    def test_roles_slimmed_to_two(self):
        """common/roles.py: 11 角色 → 2 角色 (DECOMPOSER + CONJECTURE_PROPOSER)。"""
        from common.roles import AgentRole, ROLE_PROMPTS
        assert len(AgentRole) == 2
        # 实际被删的 9 个角色不应再存在
        for dead_name in ("PROOF_GENERATOR", "PROOF_PLANNER",
                           "REPAIR_AGENT", "CRITIC",
                           "HYPOTHESIS_PROPOSER", "FORMALIZATION_EXPERT",
                           "SORRY_CLOSER", "PROOF_COMPOSER",
                           "PREMISE_RERANKER"):
            assert not hasattr(AgentRole, dead_name), (
                f"AgentRole.{dead_name} resurrection — "
                f"v13 docstring of common/roles.py records why it left.")

    def test_response_parser_slimmed(self):
        """common/response_parser.py: 仅剩 extract_lean_code."""
        import common.response_parser as rp
        assert hasattr(rp, "extract_lean_code")
        assert not hasattr(rp, "extract_json"), \
            "extract_json had 0 main-path callers in v12; do not bring it back"
        assert not hasattr(rp, "extract_sorry_blocks"), \
            "extract_sorry_blocks had 0 main-path callers in v12"

    def test_protocols_slimmed(self):
        """engine/protocols.py: 仅剩 AsyncPoolProtocol。"""
        import engine.protocols as p
        assert hasattr(p, "AsyncPoolProtocol")
        for dead in ("PoolProtocol", "VerifierProtocol", "BroadcastProtocol"):
            assert not hasattr(p, dead), (
                f"engine.protocols.{dead} had 0 type-annotation callers; "
                f"do not bring it back without an actual consumer")

    def test_persistence_back_compat_aliases_gone(self):
        """save_task_outputs / load_task_outputs / from_session_messages: 0 调用方 alias 已删。"""
        from agent.persistence import unified_storage, dialog_adapters
        assert not hasattr(unified_storage, "save_task_outputs")
        assert not hasattr(unified_storage, "load_task_outputs")
        assert not hasattr(dialog_adapters, "from_session_messages")


# ─────────────────────────────────────────────────────────────────
# C. Architecture invariants — keep the core 3 + reserved 3 alive
# ─────────────────────────────────────────────────────────────────

class TestCoreArchitecture:
    """钉项目核心定位:大一统推理 + 基础设施 + RL infra; 三个预留接口。"""

    def test_core_unified_runner_present(self):
        """大一统 profile 驱动入口必须存在。"""
        from prover.unified import UnifiedProofRunner, get_profile, PRESETS
        assert callable(UnifiedProofRunner)
        # v3 起承诺的 14 active profile 一字不少
        for name in ("whole_proof", "whole_proof_repair", "dsp", "reprover",
                      "leandojo", "heterogeneous", "conjecture_driven",
                      "kimina_batch", "pantograph_dsp", "lookeng_lemma",
                      "nfl_hybrid", "mcts", "best_first", "beam"):
            assert name in PRESETS, (
                f"core 14-profile contract broken: {name!r} missing")

    def test_core_engine_present(self):
        """基础设施 (Lean pool / backends / verification) 必须可 import。"""
        from engine import (
            AsyncLeanPool, AsyncVerificationScheduler,
            ErrorIntelligence, AgentFeedback,
        )
        assert callable(AsyncLeanPool)

    def test_rl_infra_exposed(self):
        """RL infra 接口 (vllm/slime) 必须存在。"""
        from sampler.proof_env import ProofEnv, ProofEnvConfig
        from sampler.verl_sampler import VeRLProofAgentLoop  # noqa: F401
        from sampler.slime_sampler import SlimeSampler  # noqa: F401
        assert callable(ProofEnv)

    def test_reserved_iface_knowledge(self):
        """预留接口 1: 经验知识库的管理检索与归纳总结。"""
        from knowledge.store import UnifiedKnowledgeStore  # noqa: F401
        from knowledge.reader import KnowledgeReader  # noqa: F401
        from knowledge.writer import KnowledgeWriter  # noqa: F401

    def test_reserved_iface_world_model(self):
        """预留接口 2: 世界模型。"""
        from engine.world_model import (
            WorldModelPredictor, MockWorldModel, make_world_model,
        )
        assert callable(make_world_model)

    def test_reserved_iface_broadcast(self):
        """预留接口 3: 多智能体 rollout 实时广播 (数学家 community)。"""
        from engine.broadcast import BroadcastBus, BroadcastMessage  # noqa: F401


# ─────────────────────────────────────────────────────────────────
# D. CI hygiene
# ─────────────────────────────────────────────────────────────────

def test_ci_workflow_does_not_reference_dead_test_files():
    """v13 fix: CI ``--ignore`` 路径不应再指向 v12 已重命名的测试文件。"""
    ci = ROOT / ".github/workflows/ci.yml"
    if not ci.exists():
        pytest.skip("No CI workflow file (custom checkout)")
    text = ci.read_text()
    for dead in ("test_v7_1_rl_runnable.py",
                  "test_v7_rl_unified.py",
                  "test_v6_backends_status.py"):
        assert dead not in text, (
            f"CI references {dead}, which was renamed in v12. "
            f"Update or drop the --ignore line.")

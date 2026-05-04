"""tests/test_seven_items.py — Tests for the 7 improvement items

Item 1: Legacy cleanup (LEGACY.md exists, __init__.py updated)
Item 2: World model training pipeline
Item 3: Broadcast goal-relevance sorting
Item 4: Multi-agent full role dispatch
Item 5: Knowledge TF-IDF retrieval
Item 6: Re-exports removed (direct imports work)
Item 7: Observability export

v12 note: hook_types / budget / working_memory deletions removed
the corresponding direct-import tests in TestDirectImports. All
three modules had 0 production callers; their removal is now
pinned by ``test_re_export_files_removed`` covering the shim files.
"""
import pytest
import os
import sys
import json
import tempfile

# ─── Item 6: Direct imports work ─────────────────────────────────────────────

class TestDirectImports:
    def test_common_roles(self):
        # v13: 精简到 2 个角色 (DECOMPOSER, CONJECTURE_PROPOSER) — 见
        # common/roles.py docstring。其他 9 个原本 0 主路径调用方。
        from common.roles import AgentRole, ROLE_PROMPTS
        assert len(AgentRole) == 2
        assert AgentRole.DECOMPOSER in ROLE_PROMPTS
        assert AgentRole.CONJECTURE_PROPOSER in ROLE_PROMPTS

    def test_common_response_parser(self):
        from common.response_parser import extract_lean_code
        code = extract_lean_code("Here is proof:\n```lean\n:= by simp\n```\n")
        assert "simp" in code

    def test_common_few_shot(self):
        # v13: prompt_builder 整文件删除 (109 行, 主路径只用 FEW_SHOT_EXAMPLES
        # 一个常量); 挪到 common/few_shot.py。
        from common.few_shot import FEW_SHOT_EXAMPLES
        assert isinstance(FEW_SHOT_EXAMPLES, str)
        assert "induction" in FEW_SHOT_EXAMPLES.lower()

    # v12: removed test_common_working_memory / test_common_budget /
    # test_common_hook_types — those modules had 0 production callers
    # and were deleted along with agent.memory / agent.strategy /
    # agent.hooks. The "shim removed" test below still pins the
    # cleanup invariant.

    def test_re_export_files_removed(self):
        """Verify the re-export shim files no longer exist."""
        removed = [
            "agent/brain/roles.py",
            "agent/brain/response_parser.py",
            "agent/brain/prompt_builder.py",
            "agent/hooks/hook_types.py",
            "agent/memory/working_memory.py",
            "agent/strategy/budget_allocator.py",
        ]
        for f in removed:
            assert not os.path.exists(f), f"Re-export shim should be removed: {f}"


# ─── Item 1: Legacy cleanup ─────────────────────────────────────────────────

class TestLegacyCleanup:
    """v8 起 legacy CIC 内核已彻底删除 (engine/core/, kernel/, tactic/, state/,
    search/, api/, lean_bridge/, llm/, async_search.py)。本测试组从"验证
    LEGACY.md 存在"翻转为"验证 legacy 不再存在"。
    """
    def test_legacy_dirs_removed(self):
        for d in ("core", "kernel", "tactic", "state", "search",
                  "api", "lean_bridge", "llm"):
            assert not os.path.exists(f"engine/{d}"), \
                f"engine/{d}/ should be deleted in v8"

    def test_async_search_removed(self):
        assert not os.path.exists("engine/async_search.py")

    def test_engine_init_lists_only_active_modules(self):
        with open("engine/__init__.py") as f:
            content = f.read()
        # No "Legacy modules" section anymore
        assert "Legacy modules" not in content
        # Active modules header should be there
        assert "核心模块" in content or "Active" in content


# ─── Item 2: World model training ────────────────────────────────────────────

class TestWorldModelTrainer:
    def test_trainer_imports(self):
        from engine.world_model_trainer import WorldModelTrainer, SklearnWorldModel
        assert callable(WorldModelTrainer)

    def test_trainer_no_db(self):
        from engine.world_model_trainer import WorldModelTrainer
        trainer = WorldModelTrainer(db_path="/nonexistent.db")
        n = trainer.extract_training_data()
        assert n == 0

    def test_sklearn_model_fallback(self):
        from engine.world_model_trainer import SklearnWorldModel
        model = SklearnWorldModel("/nonexistent.pkl")
        assert not model.is_trained
        # Should fall back to MockWorldModel
        pred = model.predict("⊢ n + 0 = n", "omega")
        assert hasattr(pred, "likely_success")
        assert hasattr(pred, "confidence")

    def test_extract_from_trajectories(self):
        from engine.world_model_trainer import WorldModelTrainer
        from engine.proof_context_store import StepDetail, RichProofTrajectory

        traj = RichProofTrajectory(
            theorem="theorem t : True := trivial",
            steps=[
                StepDetail(step_index=0, tactic="trivial",
                           env_id_before=0, env_id_after=1,
                           goals_before=["⊢ True"], goals_after=[]),
                StepDetail(step_index=1, tactic="ring",
                           env_id_before=1, env_id_after=-1,
                           goals_before=["⊢ n = n"], goals_after=["⊢ n = n"],
                           error_message="ring failed"),
            ],
            success=True, depth=2, duration_ms=100.0)

        trainer = WorldModelTrainer()
        n = trainer.extract_from_trajectories([traj])
        assert n == 2
        assert trainer.samples[0].success is True
        assert trainer.samples[1].success is False


# ─── Item 3: Broadcast goal relevance ────────────────────────────────────────

class TestBroadcastRelevance:
    def test_render_with_goal_relevance(self):
        from engine.broadcast import BroadcastBus, BroadcastMessage

        bus = BroadcastBus()
        sub = bus.subscribe("dir_A")

        # Publish messages with different relevance
        bus.publish(BroadcastMessage.negative(
            source="dir_B", tactic="ring", error_category="tactic_failed",
            reason="ring failed on Nat subtraction"))
        bus.publish(BroadcastMessage.positive(
            source="dir_C", discovery="omega solves Nat.add_zero directly",
            lemma_name="Nat.add_zero"))

        # Render with goal → should prioritize omega/Nat.add_zero
        text = bus.render_for_prompt(
            "dir_A", current_goal="⊢ n + 0 = n")
        assert "Teammate discoveries" in text
        assert len(text) > 0

    def test_render_without_goal(self):
        from engine.broadcast import BroadcastBus, BroadcastMessage

        bus = BroadcastBus()
        sub = bus.subscribe("dir_X")
        bus.publish(BroadcastMessage.positive(
            source="dir_Y", discovery="found useful lemma"))
        text = bus.render_for_prompt("dir_X")
        assert "found useful lemma" in text


# ─── Item 4: Full role dispatch ──────────────────────────────────────────────

# (TestFullSpectrumPlanner deleted in v9: agent/strategy/ removed entirely.
#  The 14 unified profiles in prover/unified/profiles.py replaced the old
#  light/medium/heavy/max strategy switcher.)


# ─── Item 5: TF-IDF Knowledge retrieval ─────────────────────────────────────

class TestTFIDFRetriever:
    def test_retriever_basic(self):
        from knowledge.tfidf_retriever import KnowledgeTFIDFRetriever

        retriever = KnowledgeTFIDFRetriever()
        lemmas = [
            {"name": "Nat.add_comm", "statement": "theorem Nat.add_comm (n m : Nat) : n + m = m + n", "proof": ":= by ring"},
            {"name": "Nat.mul_comm", "statement": "theorem Nat.mul_comm (n m : Nat) : n * m = m * n", "proof": ":= by ring"},
            {"name": "List.length_nil", "statement": "theorem List.length_nil : [].length = 0", "proof": ":= by rfl"},
        ]
        retriever.index_lemmas(lemmas)

        results = retriever.search("n + m = m + n", top_k=2)
        assert len(results) > 0
        # Nat.add_comm should rank higher than List.length_nil
        assert results[0].name == "Nat.add_comm"

    def test_empty_index(self):
        from knowledge.tfidf_retriever import KnowledgeTFIDFRetriever
        retriever = KnowledgeTFIDFRetriever()
        results = retriever.search("anything")
        assert results == []


# ─── Item 7: Observability export ────────────────────────────────────────────

# (TestObservabilityExport deleted in v9: engine/observability.py removed
#  -- 0 callers outside the deleted sync VerificationScheduler.
#  The async pool now uses the no-op shim engine/observability_stub.py.)

"""tests/test_seven_items.py — Tests for the 7 improvement items

Item 1: Legacy cleanup (LEGACY.md exists, __init__.py updated)
Item 2: World model training pipeline
Item 3: Broadcast goal-relevance sorting
Item 4: Multi-agent full role dispatch
Item 5: Knowledge TF-IDF retrieval
Item 6: Re-exports removed (direct imports work)
Item 7: Observability export
"""
import pytest
import os
import sys
import json
import tempfile

# ─── Item 6: Direct imports work ─────────────────────────────────────────────

class TestDirectImports:
    def test_common_roles(self):
        from common.roles import AgentRole, ROLE_PROMPTS, get_role_prompt
        assert len(AgentRole) == 11
        assert AgentRole.PROOF_GENERATOR in ROLE_PROMPTS

    def test_common_response_parser(self):
        from common.response_parser import extract_lean_code
        code = extract_lean_code("Here is proof:\n```lean\n:= by simp\n```\n")
        assert "simp" in code

    def test_common_prompt_builder(self):
        from common.prompt_builder import build_prompt
        assert callable(build_prompt)

    def test_common_working_memory(self):
        from common.working_memory import WorkingMemory
        wm = WorkingMemory()
        assert wm.solved is False

    def test_common_budget(self):
        from common.budget import Budget
        b = Budget(max_samples=10)
        assert not b.is_exhausted()

    def test_common_hook_types(self):
        from common.hook_types import HookEvent, HookAction
        assert HookEvent is not None

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
    def test_legacy_md_exists(self):
        assert os.path.exists("engine/LEGACY.md")

    def test_engine_init_marks_legacy(self):
        with open("engine/__init__.py") as f:
            content = f.read()
        assert "Legacy modules" in content
        assert "DO NOT add new dependencies" in content


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

class TestFullSpectrumPlanner:
    def test_light_strategy_base_directions(self):
        from agent.strategy.direction_planner import FullSpectrumPlanner
        from prover.models import BenchmarkProblem

        planner = FullSpectrumPlanner()
        problem = BenchmarkProblem(
            problem_id="test", name="test",
            theorem_statement="theorem t : True := by trivial",
            difficulty="easy")
        dirs = planner.plan(problem, strategy="light")
        # Light should have base directions (2-4)
        assert len(dirs) >= 2
        roles = {d.role.value for d in dirs}
        assert "proof_generator" in roles

    def test_medium_adds_decomposer(self):
        from agent.strategy.direction_planner import FullSpectrumPlanner
        from prover.models import BenchmarkProblem

        planner = FullSpectrumPlanner()
        problem = BenchmarkProblem(
            problem_id="hard", name="hard",
            theorem_statement="theorem hard " + "x" * 250,
            difficulty="hard")
        dirs = planner.plan(problem, strategy="medium",
                            attempt_history=[{"errors": [{"message": "fail"}]}])
        roles = {d.role.value for d in dirs}
        assert "decomposer" in roles
        assert "repair_agent" in roles

    def test_heavy_adds_conjecture_and_sorry(self):
        from agent.strategy.direction_planner import FullSpectrumPlanner
        from prover.models import BenchmarkProblem

        planner = FullSpectrumPlanner()
        problem = BenchmarkProblem(
            problem_id="comp", name="comp",
            theorem_statement="theorem comp " + "y" * 300,
            difficulty="competition")
        dirs = planner.plan(
            problem, strategy="heavy",
            attempt_history=[{"errors": [{}]}] * 4,
            has_sorry_skeleton=True)
        roles = {d.role.value for d in dirs}
        assert "sorry_closer" in roles
        assert "conjecture_proposer" in roles
        assert "hypothesis_proposer" in roles

    def test_max_adds_all_roles(self):
        from agent.strategy.direction_planner import FullSpectrumPlanner
        from prover.models import BenchmarkProblem

        planner = FullSpectrumPlanner()
        problem = BenchmarkProblem(
            problem_id="ultra", name="ultra",
            theorem_statement="theorem ultra " + "z" * 300,
            difficulty="competition",
            natural_language="Prove that...")
        dirs = planner.plan(
            problem, strategy="max",
            attempt_history=[{"errors": [{}]}] * 5,
            has_sorry_skeleton=True,
            banked_lemmas=["lemma h1 : True := trivial"])
        roles = {d.role.value for d in dirs}
        # Should have most roles
        assert "proof_composer" in roles
        assert "formalization_expert" in roles


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

class TestObservabilityExport:
    def test_prometheus_format(self):
        from engine.observability import MetricsCollector, MetricsExporter

        mc = MetricsCollector()
        mc.increment("proof_attempts", 42)
        mc.set_gauge("pool_size", 4.0)
        with mc.timer("verify_latency"):
            pass  # near-zero duration

        exporter = MetricsExporter(mc)
        prom = exporter.export_prometheus()
        assert "proof_attempts 42" in prom
        assert "pool_size 4" in prom
        assert "verify_latency" in prom

    def test_json_export(self):
        from engine.observability import MetricsCollector, MetricsExporter

        mc = MetricsCollector()
        mc.increment("test_counter", 7)

        exporter = MetricsExporter(mc)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            exporter.export_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "test_counter" in data
            assert data["test_counter"]["value"] == 7
            assert "_exported_at" in data
        finally:
            os.unlink(path)

    def test_snapshot_includes_all_types(self):
        from engine.observability import MetricsCollector

        mc = MetricsCollector()
        mc.increment("c1")
        mc.set_gauge("g1", 3.14)
        mc.record_time("t1", 42.0)
        snap = mc.snapshot()
        assert snap["c1"]["type"] == "counter"
        assert snap["g1"]["type"] == "gauge"
        assert snap["t1"]["type"] == "timer"
        assert snap["t1"]["p50"] == 42.0

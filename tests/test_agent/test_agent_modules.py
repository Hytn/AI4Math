"""tests/test_agent/test_agent_modules.py — Agent 基础设施模块测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytest
from agent.context.compressor import ContextCompressor
from agent.context.priority_ranker import PriorityRanker
from agent.strategy.strategy_switcher import StrategySwitcher, STRATEGIES
from agent.strategy.meta_controller import MetaController
from agent.strategy.confidence_estimator import ConfidenceEstimator
from agent.executor.resource_limiter import ResourceLimiter, ResourceLimits
from agent.executor.sandbox import Sandbox, SandboxResult
from common.working_memory import WorkingMemory
from agent.memory.episodic_memory import EpisodicMemory, Episode


# ── Context Compressor ──

class TestContextCompressor:
    def test_no_compression_needed(self):
        comp = ContextCompressor(max_tokens=10000)
        entries = [{"content": "short text", "priority": 0.5}]
        result = comp.compress(entries)
        assert len(result) == 1

    def test_truncate_strategy(self):
        comp = ContextCompressor(max_tokens=50, strategy="truncate")
        entries = [
            {"content": "A" * 100, "priority": 0.5},
            {"content": "B" * 30, "priority": 0.9},
        ]
        result = comp.compress(entries)
        # Should keep at least the last one that fits
        assert len(result) >= 1

    def test_selective_strategy(self):
        comp = ContextCompressor(max_tokens=100, strategy="selective")
        entries = [
            {"content": "low priority " * 10, "priority": 0.1},
            {"content": "high priority", "priority": 0.9},
        ]
        result = comp.compress(entries)
        # High priority should be kept
        assert any("high priority" in e["content"] for e in result)

    def test_summarize_strategy(self):
        comp = ContextCompressor(max_tokens=100, strategy="summarize")
        entries = [{"content": f"entry {i}" * 5, "priority": 0.5} for i in range(10)]
        result = comp.compress(entries)
        assert len(result) < len(entries)
        # Should have a summary entry
        assert any(e.get("is_summary") for e in result)

    def test_compress_text(self):
        comp = ContextCompressor(max_tokens=50)
        long_text = "\n".join(f"line {i}" for i in range(100))
        result = comp.compress_text(long_text)
        assert len(result) < len(long_text)
        assert "omitted" in result

    def test_empty_entries(self):
        comp = ContextCompressor()
        assert comp.compress([]) == []


# ── Priority Ranker ──

class TestPriorityRanker:
    def test_rank_by_category(self):
        ranker = PriorityRanker()
        items = [
            {"content": "goal", "category": "current_goal"},
            {"content": "meta", "category": "metadata"},
            {"content": "theorem", "category": "theorem_statement"},
        ]
        ranked = ranker.rank(items)
        assert ranked[0].category == "theorem_statement"
        assert ranked[-1].category == "metadata"

    def test_filter_by_budget(self):
        ranker = PriorityRanker()
        items = [
            {"content": "A" * 300, "category": "metadata"},
            {"content": "B" * 30, "category": "theorem_statement"},
        ]
        result = ranker.filter_by_budget(items, token_budget=50)
        # Should keep the high-priority short item
        assert len(result) >= 1

    def test_custom_priorities(self):
        ranker = PriorityRanker({"custom_cat": 0.99})
        items = [{"content": "test", "category": "custom_cat"}]
        ranked = ranker.rank(items)
        assert ranked[0].priority >= 0.99


# ── Strategy Switcher ──

class TestStrategySwitcher:
    def test_switch_escalation(self):
        assert StrategySwitcher.switch("light", "medium") == "medium"
        assert StrategySwitcher.switch("medium", "heavy") == "heavy"
        assert StrategySwitcher.switch("heavy", "heavy") == "heavy"

    def test_switch_to_unknown(self):
        result = StrategySwitcher.switch("light", "nonexistent")
        assert result in STRATEGIES  # should escalate, not crash

    def test_get_config(self):
        cfg = StrategySwitcher.get_config("heavy")
        assert cfg.name == "heavy"
        assert cfg.samples_per_round >= 8
        assert cfg.use_conjecture is True

    def test_get_config_unknown(self):
        cfg = StrategySwitcher.get_config("nonexistent")
        assert cfg.name == "light"  # fallback

    def test_escalation_path(self):
        path = StrategySwitcher.get_escalation_path("light")
        assert path == ["medium", "heavy"]

    def test_available_strategies(self):
        available = StrategySwitcher.available_strategies()
        assert "light" in available
        assert "heavy" in available
        assert len(available) >= 4

    def test_config_consistency(self):
        for name, cfg in STRATEGIES.items():
            assert cfg.name == name
            assert cfg.samples_per_round > 0
            assert cfg.max_rounds > 0


# ── Meta Controller ──

class TestMetaController:
    def test_initial_strategy_easy(self):
        mc = MetaController()
        assert mc.select_initial_strategy("easy") == "sequential"

    def test_initial_strategy_hard(self):
        mc = MetaController()
        assert mc.select_initial_strategy("hard") == "medium"

    def test_escalation(self):
        mc = MetaController({"max_light_rounds": 2})
        # BudgetEscalationRule triggers when total_samples/max_samples > 0.3
        mem = WorkingMemory(current_strategy="light", rounds_completed=3,
                            total_samples=50)  # 50/128 ≈ 39% > 30% threshold
        assert mc.should_escalate(mem) == "medium"

    def test_no_escalation_if_solved(self):
        mc = MetaController()
        mem = WorkingMemory(current_strategy="light", rounds_completed=100, solved=True)
        assert mc.should_escalate(mem) is None


# ── Confidence Estimator ──

class TestConfidenceEstimator:
    def test_solved_full_confidence(self):
        ce = ConfidenceEstimator()
        mem = WorkingMemory(solved=True)
        assert ce.estimate(mem) == 1.0

    def test_no_attempts_half(self):
        ce = ConfidenceEstimator()
        mem = WorkingMemory()
        assert ce.estimate(mem) == 0.5

    def test_should_abstain_low(self):
        ce = ConfidenceEstimator()
        mem = WorkingMemory(total_samples=200)
        # After many failed attempts, confidence drops
        for _ in range(50):
            mem.record_attempt({"errors": [{"category": "tactic_failed"}]})
        assert ce.should_abstain(mem)

    def test_should_not_abstain_fresh(self):
        ce = ConfidenceEstimator()
        mem = WorkingMemory()
        assert not ce.should_abstain(mem)


# ── Resource Limiter ──

class TestResourceLimiter:
    def test_not_exceeded_initially(self):
        limiter = ResourceLimiter(ResourceLimits(timeout_seconds=60))
        limiter.start()
        assert not limiter.is_exceeded()

    def test_api_call_limit(self):
        limiter = ResourceLimiter(ResourceLimits(max_api_calls=2))
        limiter.start()
        limiter.record_api_call(100)
        limiter.record_api_call(100)
        assert limiter.is_exceeded()

    def test_token_limit(self):
        limiter = ResourceLimiter(ResourceLimits(max_tokens=100))
        limiter.start()
        limiter.record_api_call(tokens=150)
        assert limiter.is_exceeded()

    def test_remaining_budget(self):
        limiter = ResourceLimiter(ResourceLimits(max_api_calls=10))
        limiter.start()
        limiter.record_api_call(50)
        budget = limiter.remaining_budget()
        assert budget["api_calls"] == 9

    def test_utilization(self):
        limiter = ResourceLimiter(ResourceLimits(max_api_calls=10))
        limiter.start()
        for _ in range(5):
            limiter.record_api_call()
        util = limiter.utilization()
        assert util["api_calls"] == 0.5


# ── Sandbox ──

class TestSandbox:
    def test_allowed_command(self):
        sandbox = Sandbox(allowed_commands=["echo"])
        result = sandbox.run(["echo", "hello"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_blocked_command(self):
        sandbox = Sandbox(allowed_commands=["echo"])
        result = sandbox.run(["rm", "-rf", "/"])
        assert result.returncode == -1
        assert "not in allowed" in result.stderr

    def test_timeout(self):
        sandbox = Sandbox(allowed_commands=["sleep"])
        result = sandbox.run(["sleep", "10"], timeout=1)
        assert result.timed_out

    def test_command_not_found(self):
        sandbox = Sandbox(allowed_commands=["nonexistent_cmd_xyz"])
        result = sandbox.run(["nonexistent_cmd_xyz"])
        assert result.returncode == -1


# ── Episodic Memory ──

class TestEpisodicMemory:
    def test_add_and_retrieve(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as f:
            mem = EpisodicMemory(store_path=f.name)
            mem.episodes = []  # reset
            ep = Episode("algebra", "easy", "ring", ["ring", "simp"], "use ring", 500)
            mem.add(ep)
            assert len(mem.episodes) == 1

            similar = mem.retrieve_similar("algebra")
            assert len(similar) == 1
            assert similar[0].winning_strategy == "ring"

    def test_retrieve_empty(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True) as f:
            mem = EpisodicMemory(store_path=f.name)
            mem.episodes = []
            assert mem.retrieve_similar("anything") == []

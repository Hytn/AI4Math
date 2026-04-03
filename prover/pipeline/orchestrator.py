"""prover/pipeline/orchestrator.py — 主调度器"""
from __future__ import annotations
import logging
from prover.models import BenchmarkProblem, ProofTrace
from prover.pipeline.rollout_engine import RolloutEngine
from prover.pipeline.sequential_engine import SequentialEngine
from agent.strategy.meta_controller import MetaController
from agent.strategy.strategy_switcher import StrategySwitcher
from agent.strategy.reflection import Reflector
from agent.strategy.budget_allocator import Budget
from agent.strategy.confidence_estimator import ConfidenceEstimator
from agent.memory.working_memory import WorkingMemory
from agent.context.context_window import ContextWindow

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, lean_env, llm_provider, retriever=None, config=None, on_attempt=None):
        self.lean = lean_env; self.llm = llm_provider; self.retriever = retriever
        self.config = config or {}; self.on_attempt = on_attempt
        self.meta = MetaController(config)
        self.reflector = Reflector(llm_provider)
        self.confidence = ConfidenceEstimator()
        self.budget = Budget(max_samples=config.get("max_samples", 128))

    def prove(self, problem: BenchmarkProblem) -> ProofTrace:
        memory = WorkingMemory(problem_id=problem.problem_id,
                               theorem_statement=problem.theorem_statement)
        ctx = ContextWindow()
        strategy = self.meta.select_initial_strategy(problem.difficulty)
        memory.current_strategy = strategy
        trace = ProofTrace(problem_id=problem.problem_id, problem_name=problem.name,
                           theorem_statement=problem.theorem_statement,
                           natural_language=problem.natural_language)

        while not memory.solved and not self.budget.is_exhausted():
            if self.confidence.should_abstain(memory):
                logger.info(f"Abstaining from {problem.name} — low confidence"); break
            escalation = self.meta.should_escalate(memory)
            if escalation:
                strategy = StrategySwitcher.switch(strategy, escalation)
                memory.current_strategy = strategy
                trace.strategy_path.append(strategy)

            if strategy == "sequential":
                engine = SequentialEngine(self.lean, self.llm, self.retriever, self.config)
            else:
                engine = RolloutEngine(self.lean, self.llm, self.retriever, self.config)

            round_trace = engine.run_round(problem, memory, self.budget)
            for a in round_trace: trace.add_attempt(a)
            memory.rounds_completed += 1

            if trace.solved: break

        return trace

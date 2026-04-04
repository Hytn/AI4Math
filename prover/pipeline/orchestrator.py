"""prover/pipeline/orchestrator.py — 主调度器"""
from __future__ import annotations
import logging
import time
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
    def __init__(self, lean_env, llm_provider, retriever=None,
                 config=None, on_attempt=None):
        self.lean = lean_env
        self.llm = llm_provider
        self.retriever = retriever
        self.config = config or {}
        self.on_attempt = on_attempt
        self.meta = MetaController(self.config)
        self.reflector = Reflector(llm_provider)
        self.confidence = ConfidenceEstimator()
        self.budget = Budget(
            max_samples=self.config.get("max_samples", 128),
            max_wall_seconds=self.config.get("max_wall_seconds", 3600),
        )

    def prove(self, problem: BenchmarkProblem) -> ProofTrace:
        start_time = time.time()

        memory = WorkingMemory(
            problem_id=problem.problem_id,
            theorem_statement=problem.theorem_statement)
        ctx = ContextWindow()
        strategy_name = self.meta.select_initial_strategy(problem.difficulty)
        memory.current_strategy = strategy_name

        trace = ProofTrace(
            problem_id=problem.problem_id,
            problem_name=problem.name,
            theorem_statement=problem.theorem_statement,
            natural_language=problem.natural_language,
            config_snapshot={
                "strategy": strategy_name,
                "max_samples": self.budget.max_samples,
            })

        while not memory.solved and not self.budget.is_exhausted():
            # Confidence-based early stopping
            if self.confidence.should_abstain(memory):
                logger.info(
                    f"Abstaining from {problem.name} — low confidence")
                break

            # Strategy escalation check
            escalation = self.meta.should_escalate(memory)
            if escalation:
                old_strategy = strategy_name
                strategy_name = StrategySwitcher.switch(
                    strategy_name, escalation)
                memory.current_strategy = strategy_name
                trace.strategy_path.append(strategy_name)
                logger.info(
                    f"Strategy escalation: {old_strategy} → {strategy_name}")

                # On escalation, run reflection to analyze failures
                if memory.attempt_history:
                    self._run_reflection(problem, memory)

            # Get strategy configuration
            strategy_config = StrategySwitcher.get_config(strategy_name)

            # Build engine config from strategy
            engine_config = {
                **self.config,
                "samples_per_round": strategy_config.samples_per_round,
                "max_workers": strategy_config.max_workers,
                "temperature": strategy_config.temperature,
                "max_repair_rounds": 3 if strategy_config.use_repair else 0,
                "max_attempts": strategy_config.samples_per_round,
            }

            # Select engine type
            if strategy_name == "sequential":
                engine = SequentialEngine(
                    self.lean, self.llm, self.retriever, engine_config)
            else:
                engine = RolloutEngine(
                    self.lean, self.llm, self.retriever, engine_config)

            # Decompose if strategy supports it and we've failed enough
            if (strategy_config.use_decompose
                    and memory.rounds_completed >= 2
                    and not memory.solved):
                self._try_decompose(problem, memory)

            # Run proof round
            round_trace = engine.run_round(problem, memory, self.budget)
            for a in round_trace:
                trace.add_attempt(a)
                # Track tokens in budget
                if hasattr(a, 'llm_tokens_in'):
                    self.budget.add_tokens(
                        a.llm_tokens_in + a.llm_tokens_out)
                if self.on_attempt:
                    self.on_attempt(a)
            memory.rounds_completed += 1

            if trace.solved:
                break

            # Conjecture generation for heavy strategy
            if (strategy_config.use_conjecture
                    and not memory.solved
                    and memory.rounds_completed >= 3):
                self._try_conjecture(problem, memory)

        trace.total_duration_ms = int((time.time() - start_time) * 1000)
        return trace

    def _run_reflection(self, problem, memory):
        """Run reflection to analyze failures and adjust strategy."""
        try:
            error_summary = memory.get_dominant_error()
            best_proofs = [
                a.get("generated_proof", "")[:200]
                for a in memory.attempt_history[-3:]
                if a.get("generated_proof")
            ]
            reflection = self.reflector.reflect(
                problem.theorem_statement, error_summary, best_proofs)
            logger.info(f"Reflection: {reflection[:200]}")
        except Exception as e:
            logger.debug(f"Reflection failed: {e}")

    def _try_decompose(self, problem, memory):
        """Attempt goal decomposition when direct proving fails."""
        try:
            from prover.decompose.goal_decomposer import GoalDecomposer
            decomposer = GoalDecomposer(self.llm)
            subgoals = decomposer.decompose(problem.theorem_statement)
            if subgoals:
                logger.info(
                    f"Decomposed into {len(subgoals)} sub-goals")
                for sg in subgoals:
                    memory.goal_stack.append(sg.statement)
        except Exception as e:
            logger.debug(f"Decomposition failed: {e}")

    def _try_conjecture(self, problem, memory):
        """Generate auxiliary conjectures for hard problems."""
        try:
            from prover.conjecture.conjecture_proposer import ConjectureProposer
            proposer = ConjectureProposer(self.llm)
            existing = [
                l.get("statement", "")
                for l in memory.banked_lemmas[:5]
            ]
            conjectures = proposer.propose(
                problem.theorem_statement,
                existing_lemmas=existing,
                n=3, verify=False)
            if conjectures:
                logger.info(
                    f"Generated {len(conjectures)} conjectures")
                for conj in conjectures:
                    memory.banked_lemmas.append({
                        "name": "conj",
                        "statement": conj,
                        "proof": "",
                        "verified": False,
                    })
        except Exception as e:
            logger.debug(f"Conjecture generation failed: {e}")

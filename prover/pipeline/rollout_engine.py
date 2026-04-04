"""prover/pipeline/rollout_engine.py — 并行 Rollout 引擎"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from prover.models import BenchmarkProblem, ProofAttempt, AttemptStatus
from prover.pipeline.proof_loop import ProofLoop
from agent.memory.working_memory import WorkingMemory
from agent.strategy.budget_allocator import Budget
from agent.context.error_summarizer import summarize_round_errors

logger = logging.getLogger(__name__)

class RolloutEngine:
    def __init__(self, lean_env, llm, retriever=None, config=None):
        self.lean = lean_env; self.llm = llm; self.retriever = retriever
        self.config = config or {}
        self.samples_per_round = self.config.get("samples_per_round", 8)
        self.max_workers = self.config.get("max_workers", 4)
        self.temperature = self.config.get("temperature", 0.9)

    def run_round(self, problem: BenchmarkProblem, memory: WorkingMemory,
                  budget: Budget) -> list[ProofAttempt]:
        loop = ProofLoop(self.lean, self.llm, self.retriever, self.config)
        attempts = []

        # Snapshot the current sample count for indexing
        base_idx = budget.samples_used

        def _sample(idx):
            return loop.single_attempt(problem, memory, temperature=self.temperature,
                                       attempt_num=base_idx + idx + 1)

        workers = min(self.max_workers, self.samples_per_round)
        if workers <= 1:
            for i in range(self.samples_per_round):
                attempts.append(_sample(i))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_sample, i): i for i in range(self.samples_per_round)}
                for f in as_completed(futures):
                    try:
                        attempts.append(f.result())
                    except Exception as e:
                        logger.error(f"Sample error: {e}")

        # Thread-safe budget update (single call after all threads complete)
        budget.add_samples(len(attempts))

        # Thread-safe memory updates and token tracking (sequential after collection)
        total_tokens = 0
        for a in attempts:
            total_tokens += a.llm_tokens_in + a.llm_tokens_out
            memory.record_attempt(
                a.to_dict() if hasattr(a, 'to_dict') else {"errors": []})
            if a.lean_result == AttemptStatus.SUCCESS:
                memory.solved = True

        # Track total tokens used this round
        budget.add_tokens(total_tokens)

        return attempts

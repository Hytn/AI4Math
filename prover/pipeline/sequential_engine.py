"""prover/pipeline/sequential_engine.py — 顺序重试引擎"""
from __future__ import annotations
from prover.models import BenchmarkProblem, ProofAttempt, AttemptStatus
from prover.pipeline.proof_loop import ProofLoop
from common.working_memory import WorkingMemory
from common.budget import Budget

class SequentialEngine:
    def __init__(self, lean_env, llm, retriever=None, config=None):
        self.lean = lean_env; self.llm = llm; self.retriever = retriever
        self.config = config or {}
        self.max_attempts = self.config.get("max_attempts", 10)

    def run_round(self, problem: BenchmarkProblem, memory: WorkingMemory,
                  budget: Budget) -> list[ProofAttempt]:
        loop = ProofLoop(self.lean, self.llm, self.retriever, self.config)
        attempts = []
        for i in range(self.max_attempts):
            if budget.is_exhausted():
                break
            a = loop.single_attempt(problem, memory,
                                     attempt_num=budget.samples_used + 1)
            attempts.append(a)
            budget.add_samples(1)
            memory.record_attempt(
                a.to_dict() if hasattr(a, 'to_dict') else {"errors": []})
            if a.lean_result == AttemptStatus.SUCCESS:
                memory.solved = True
                break
        return attempts

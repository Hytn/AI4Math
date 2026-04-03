"""agent/strategy/confidence_estimator.py — 置信度评估与主动放弃"""
from __future__ import annotations
from agent.memory.working_memory import WorkingMemory

class ConfidenceEstimator:
    def estimate(self, memory: WorkingMemory) -> float:
        if memory.solved: return 1.0
        if not memory.attempt_history: return 0.5
        n = len(memory.attempt_history)
        dom = memory.get_dominant_error()
        progress = len(memory.banked_lemmas) / max(1, n) * 10
        repetition_penalty = 0.1 * sum(1 for a in memory.attempt_history[-5:]
                                        if dom in str(a.get("errors", [])))
        return max(0.0, min(1.0, 0.5 + progress - repetition_penalty - n * 0.01))

    def should_abstain(self, memory: WorkingMemory, threshold: float = 0.1) -> bool:
        return self.estimate(memory) < threshold

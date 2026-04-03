"""agent/strategy/meta_controller.py — 元控制器: 根据当前状态选择策略"""
from __future__ import annotations
from agent.memory.working_memory import WorkingMemory

class MetaController:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_light_rounds = self.config.get("max_light_rounds", 2)
        self.max_medium_rounds = self.config.get("max_medium_rounds", 4)

    def select_initial_strategy(self, difficulty: str) -> str:
        if difficulty in ("easy", "trivial"): return "sequential"
        return "light"

    def should_escalate(self, memory: WorkingMemory) -> str | None:
        if memory.solved: return None
        if memory.current_strategy == "light" and memory.rounds_completed >= self.max_light_rounds:
            return "medium"
        if memory.current_strategy == "medium" and memory.rounds_completed >= self.max_medium_rounds:
            return "heavy"
        return None

    def should_give_up(self, memory: WorkingMemory, budget: dict) -> bool:
        if memory.total_samples >= budget.get("max_samples", 128):
            return True
        return False

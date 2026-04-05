"""agent/memory/working_memory.py — 工作记忆: 当前证明任务状态"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field

@dataclass
class WorkingMemory:
    problem_id: str = ""
    theorem_statement: str = ""
    goal_stack: list[str] = field(default_factory=list)
    attempt_history: list[dict] = field(default_factory=list)
    error_patterns: dict[str, int] = field(default_factory=dict)
    banked_lemmas: list[dict] = field(default_factory=list)
    current_strategy: str = "light"
    rounds_completed: int = 0
    total_samples: int = 0
    solved: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_attempt(self, attempt: dict):
        with self._lock:
            self.attempt_history.append(attempt)
            self.total_samples += 1
            for err in attempt.get("errors", []):
                cat = err.get("category", "other")
                self.error_patterns[cat] = self.error_patterns.get(cat, 0) + 1

    def get_dominant_error(self) -> str:
        with self._lock:
            if not self.error_patterns:
                return "none"
            return max(self.error_patterns, key=self.error_patterns.get)

    # ── 序列化支持 (pickle/deepcopy) ──
    # threading.Lock 不可序列化, 需自定义 __getstate__/__setstate__

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

"""agent/strategy/budget_allocator.py — 预算分配"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Budget:
    max_samples: int = 128
    max_tokens: int = 2_000_000
    max_wall_seconds: int = 3600
    samples_used: int = 0
    tokens_used: int = 0

    def remaining_samples(self) -> int:
        return max(0, self.max_samples - self.samples_used)

    def is_exhausted(self) -> bool:
        return self.samples_used >= self.max_samples

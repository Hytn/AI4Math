"""common/budget.py — Resource budget tracking (shared)"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_samples: int = 128
    max_tokens: int = 2_000_000
    max_wall_seconds: int = 3600
    samples_used: int = 0
    tokens_used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _start_time: float = field(default_factory=time.monotonic, repr=False)

    def remaining_samples(self) -> int:
        with self._lock:
            return max(0, self.max_samples - self.samples_used)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def is_exhausted(self) -> bool:
        with self._lock:
            if self.samples_used >= self.max_samples:
                return True
            if self.tokens_used >= self.max_tokens:
                return True
        # Wall-time check (no lock needed, monotonic)
        if self.elapsed_seconds() >= self.max_wall_seconds:
            return True
        return False

    def add_samples(self, n: int):
        """Thread-safe increment of samples_used."""
        with self._lock:
            self.samples_used += n

    def add_tokens(self, n: int):
        """Thread-safe increment of tokens_used."""
        with self._lock:
            self.tokens_used += n

    def summary(self) -> dict:
        """Return budget usage summary."""
        with self._lock:
            return {
                "samples_used": self.samples_used,
                "max_samples": self.max_samples,
                "tokens_used": self.tokens_used,
                "max_tokens": self.max_tokens,
                "elapsed_seconds": round(self.elapsed_seconds(), 1),
                "max_wall_seconds": self.max_wall_seconds,
            }

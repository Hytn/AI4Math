"""agent/executor/resource_limiter.py — 资源限制与监控

跟踪和限制计算资源使用: 时间、内存、API 调用。
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class ResourceLimits:
    """Resource limits for a single proof attempt."""
    timeout_seconds: int = 120
    max_memory_mb: int = 4096
    max_concurrent: int = 4
    max_api_calls: int = 100
    max_tokens: int = 500_000


@dataclass
class ResourceUsage:
    """Tracked resource usage."""
    elapsed_seconds: float = 0.0
    api_calls: int = 0
    tokens_used: int = 0
    peak_memory_mb: float = 0.0
    lean_compilations: int = 0


class ResourceLimiter:
    """Monitor and enforce resource limits.

    Usage:
        limiter = ResourceLimiter(ResourceLimits(timeout_seconds=60))
        limiter.start()
        ...
        if limiter.is_exceeded():
            break
        limiter.record_api_call(tokens=500)
    """

    def __init__(self, limits: ResourceLimits = None):
        self.limits = limits or ResourceLimits()
        self.usage = ResourceUsage()
        self._start_time: float = 0.0

    def start(self):
        """Start resource tracking."""
        self._start_time = time.time()

    def is_exceeded(self) -> bool:
        """Check if any resource limit has been exceeded."""
        self._update_elapsed()

        if self.usage.elapsed_seconds > self.limits.timeout_seconds:
            return True
        if self.usage.api_calls >= self.limits.max_api_calls:
            return True
        if self.usage.tokens_used >= self.limits.max_tokens:
            return True
        return False

    def record_api_call(self, tokens: int = 0):
        """Record an API call."""
        self.usage.api_calls += 1
        self.usage.tokens_used += tokens

    def record_lean_compilation(self):
        """Record a Lean compilation."""
        self.usage.lean_compilations += 1

    def remaining_budget(self) -> dict:
        """Get remaining budget for each resource."""
        self._update_elapsed()
        return {
            "time_seconds": max(0, self.limits.timeout_seconds - self.usage.elapsed_seconds),
            "api_calls": max(0, self.limits.max_api_calls - self.usage.api_calls),
            "tokens": max(0, self.limits.max_tokens - self.usage.tokens_used),
        }

    def utilization(self) -> dict:
        """Get utilization percentage for each resource."""
        self._update_elapsed()
        return {
            "time": self.usage.elapsed_seconds / max(1, self.limits.timeout_seconds),
            "api_calls": self.usage.api_calls / max(1, self.limits.max_api_calls),
            "tokens": self.usage.tokens_used / max(1, self.limits.max_tokens),
        }

    def _update_elapsed(self):
        if self._start_time > 0:
            self.usage.elapsed_seconds = time.time() - self._start_time

"""engine/observability_stub.py — No-op metrics shim.

Replaces ``engine/observability.py`` (deleted in v9 cleanup, 0
non-internal callers). Keeps the call sites in ``async_lean_pool.py``
and ``async_verification_scheduler.py`` working without rewriting them.

If/when real metrics are needed, replace this with a Prometheus or
OpenTelemetry exporter and wire it into ``run_unified.py``.
"""
from __future__ import annotations
from contextlib import contextmanager


class _NoOpMetrics:
    """Drop-in replacement for the old ``metrics`` module."""

    def increment(self, key: str, by: int = 1) -> None:
        pass

    def record_time(self, key: str, ms: float) -> None:
        pass

    def gauge(self, key: str, value: float) -> None:
        pass

    @contextmanager
    def timer(self, key: str):
        # No-op timer: enter / exit do nothing.
        yield

    def snapshot(self) -> dict:
        return {}


metrics = _NoOpMetrics()

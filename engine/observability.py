"""engine/observability.py — 可观测性基础设施

提供结构化日志和指标收集, 为"毫秒级响应"目标提供度量基础。

三大能力:
  1. 结构化日志:  每条日志带 context_id / problem_id / direction 等字段
  2. 指标收集:    计时器 + 计数器 + 直方图, 可接入 Prometheus / OTEL
  3. Trace 增强:  ProofTrace 记录每个 stage 的耗时分解

设计原则:
  - 零开销 when disabled (通过 enabled flag 控制)
  - 线程安全 (指标通过 threading.Lock 保护)
  - 可插拔 (MetricsBackend 协议, 默认 in-memory, 可替换为 Prometheus)

Usage::

    from engine.observability import metrics, timed

    # 手动埋点
    with metrics.timer("repl_verify_latency", direction="structured"):
        result = pool.verify_complete(theorem, proof)

    # 装饰器埋点
    @timed("llm_generate")
    def generate(self, prompt): ...

    # 查询指标
    stats = metrics.snapshot()
    print(stats["repl_verify_latency"]["p50"])
"""
from __future__ import annotations
import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 结构化日志
# ═══════════════════════════════════════════════════════════════

class StructuredLogger:
    """带上下文字段的结构化日志包装器

    自动附加 problem_id / direction / session_id 等字段。

    Usage::

        slog = StructuredLogger("engine.lean_pool")
        slog = slog.bind(problem_id="minif2f_001", direction="automation")
        slog.info("tactic_executed", tactic="simp", elapsed_ms=42)
        # → [engine.lean_pool] tactic_executed problem_id=minif2f_001 direction=automation tactic=simp elapsed_ms=42
    """

    def __init__(self, name: str, **context):
        self._logger = logging.getLogger(name)
        self._context = context

    def bind(self, **kwargs) -> 'StructuredLogger':
        """创建带额外上下文的新 logger (不修改原 logger)"""
        merged = {**self._context, **kwargs}
        new = StructuredLogger(self._logger.name, **merged)
        new._logger = self._logger
        return new

    def _format(self, event: str, **kwargs) -> str:
        all_fields = {**self._context, **kwargs}
        parts = [event]
        for k, v in all_fields.items():
            parts.append(f"{k}={v}")
        return " ".join(parts)

    def debug(self, event: str, **kwargs):
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(self._format(event, **kwargs))

    def info(self, event: str, **kwargs):
        self._logger.info(self._format(event, **kwargs))

    def warning(self, event: str, **kwargs):
        self._logger.warning(self._format(event, **kwargs))

    def error(self, event: str, **kwargs):
        self._logger.error(self._format(event, **kwargs))


# ═══════════════════════════════════════════════════════════════
# 指标收集
# ═══════════════════════════════════════════════════════════════

@dataclass
class TimerSample:
    """单次计时样本"""
    name: str
    duration_ms: float
    labels: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """线程安全的指标收集器

    支持:
      - timer: 记录耗时 (直方图)
      - counter: 记录次数
      - gauge: 记录当前值

    所有指标按 name + labels 分组。
    """

    def __init__(self, enabled: bool = True, max_samples: int = 10000):
        self._enabled = enabled
        self._max_samples = max_samples
        self._lock = threading.Lock()

        # 计时样本 (name → [duration_ms, ...])
        self._timers: dict[str, list[float]] = defaultdict(list)
        # 计数器 (name → count)
        self._counters: dict[str, int] = defaultdict(int)
        # 仪表盘 (name → current_value)
        self._gauges: dict[str, float] = {}

    @contextmanager
    def timer(self, name: str, **labels):
        """计时上下文管理器

        Usage::

            with metrics.timer("repl_latency", level="L1"):
                result = pool.verify(...)
        """
        if not self._enabled:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            key = self._make_key(name, labels)
            with self._lock:
                samples = self._timers[key]
                samples.append(elapsed_ms)
                # 防止内存泄漏
                if len(samples) > self._max_samples:
                    self._timers[key] = samples[-self._max_samples:]

    def record_time(self, name: str, duration_ms: float, **labels):
        """手动记录耗时"""
        if not self._enabled:
            return
        key = self._make_key(name, labels)
        with self._lock:
            samples = self._timers[key]
            samples.append(duration_ms)
            if len(samples) > self._max_samples:
                self._timers[key] = samples[-self._max_samples:]

    def increment(self, name: str, delta: int = 1, **labels):
        """递增计数器"""
        if not self._enabled:
            return
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += delta

    def set_gauge(self, name: str, value: float, **labels):
        """设置仪表盘值"""
        if not self._enabled:
            return
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def snapshot(self) -> dict:
        """获取所有指标的快照"""
        with self._lock:
            result = {}

            for key, samples in self._timers.items():
                if not samples:
                    continue
                sorted_s = sorted(samples)
                n = len(sorted_s)
                result[key] = {
                    "type": "timer",
                    "count": n,
                    "min": round(sorted_s[0], 2),
                    "max": round(sorted_s[-1], 2),
                    "mean": round(sum(sorted_s) / n, 2),
                    "p50": round(sorted_s[n // 2], 2),
                    "p90": round(sorted_s[int(n * 0.9)], 2),
                    "p99": round(sorted_s[int(n * 0.99)], 2),
                }

            for key, count in self._counters.items():
                result[key] = {"type": "counter", "value": count}

            for key, value in self._gauges.items():
                result[key] = {"type": "gauge", "value": value}

            return result

    def reset(self):
        """重置所有指标"""
        with self._lock:
            self._timers.clear()
            self._counters.clear()
            self._gauges.clear()

    @staticmethod
    def _make_key(name: str, labels: dict) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# ═══════════════════════════════════════════════════════════════
# 全局实例 + 装饰器
# ═══════════════════════════════════════════════════════════════

# 全局指标收集器 (可通过 configure() 替换)
metrics = MetricsCollector(enabled=True)


def timed(metric_name: str, **default_labels):
    """计时装饰器

    Usage::

        @timed("llm_generate")
        def generate(self, prompt):
            ...

        @timed("verify", level="L1")
        async def verify_tactic(self, env_id, tactic):
            ...
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics.timer(metric_name, **default_labels):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - t0) * 1000
                metrics.record_time(metric_name, elapsed, **default_labels)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def configure(enabled: bool = True, max_samples: int = 10000):
    """重新配置全局指标收集器"""
    global metrics
    metrics = MetricsCollector(enabled=enabled, max_samples=max_samples)

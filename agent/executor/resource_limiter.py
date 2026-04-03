"""agent/executor/resource_limiter.py — 资源限制"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ResourceLimits:
    timeout_seconds: int = 120
    max_memory_mb: int = 4096
    max_concurrent: int = 4

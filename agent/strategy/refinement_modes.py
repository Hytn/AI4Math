"""agent/strategy/refinement_modes.py — Light / Medium / Heavy 推理模式"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LightConfig:
    samples_per_round: int = 16
    max_rounds: int = 2
    temperature: float = 0.9
    max_repair_per_sample: int = 3

@dataclass
class MediumConfig:
    outer_samples: int = 4
    inner_samples: int = 8
    max_rounds: int = 4
    enable_decompose: bool = True
    enable_inner_refinement: bool = True

@dataclass
class HeavyConfig:
    conjecture_rounds: int = 10
    samples_per_round: int = 16
    max_rounds: int = 8
    enable_conjecture_pool: bool = True
    enable_cas: bool = True
    max_wall_hours: int = 24

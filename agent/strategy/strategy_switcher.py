"""agent/strategy/strategy_switcher.py — 策略切换器

管理从 Light → Medium → Heavy 的策略升级，以及参数调整。
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """Configuration for a proving strategy level."""
    name: str
    samples_per_round: int
    max_rounds: int
    temperature: float
    max_workers: int
    use_repair: bool
    use_decompose: bool
    use_conjecture: bool
    max_search_depth: int = 10
    description: str = ""


# Pre-defined strategy levels
STRATEGIES: dict[str, StrategyConfig] = {
    "sequential": StrategyConfig(
        name="sequential", samples_per_round=1, max_rounds=3,
        temperature=0.3, max_workers=1,
        use_repair=False, use_decompose=False, use_conjecture=False,
        description="Simple sequential attempts, low temperature",
    ),
    "light": StrategyConfig(
        name="light", samples_per_round=4, max_rounds=3,
        temperature=0.7, max_workers=2,
        use_repair=True, use_decompose=False, use_conjecture=False,
        description="Parallel sampling with repair, moderate temperature",
    ),
    "medium": StrategyConfig(
        name="medium", samples_per_round=8, max_rounds=5,
        temperature=0.9, max_workers=4,
        use_repair=True, use_decompose=True, use_conjecture=False,
        max_search_depth=20,
        description="Full parallelism with decomposition and repair",
    ),
    "heavy": StrategyConfig(
        name="heavy", samples_per_round=16, max_rounds=10,
        temperature=1.0, max_workers=4,
        use_repair=True, use_decompose=True, use_conjecture=True,
        max_search_depth=50,
        description="Maximum effort: conjecture generation, deep search",
    ),
}


class StrategySwitcher:
    """Manage strategy transitions."""

    @staticmethod
    def switch(current: str, target: str) -> str:
        """Switch from current strategy to target.

        Returns the new strategy name (validated).
        """
        if target in STRATEGIES:
            return target
        # Escalation: light → medium → heavy
        escalation = {"sequential": "light", "light": "medium",
                       "medium": "heavy", "heavy": "heavy"}
        return escalation.get(current, "medium")

    @staticmethod
    def get_config(strategy_name: str) -> StrategyConfig:
        """Get configuration for a named strategy."""
        return STRATEGIES.get(strategy_name, STRATEGIES["light"])

    @staticmethod
    def get_escalation_path(current: str) -> list[str]:
        """Get the remaining escalation path from current strategy."""
        order = ["sequential", "light", "medium", "heavy"]
        try:
            idx = order.index(current)
            return order[idx + 1:]
        except ValueError:
            return ["medium", "heavy"]

    @staticmethod
    def available_strategies() -> list[str]:
        return list(STRATEGIES.keys())

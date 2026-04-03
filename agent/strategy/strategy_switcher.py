"""agent/strategy/strategy_switcher.py — 策略切换"""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

class StrategySwitcher:
    VALID = ["sequential", "light", "medium", "heavy"]

    @staticmethod
    def switch(current: str, target: str) -> str:
        if target not in StrategySwitcher.VALID:
            raise ValueError(f"Invalid strategy: {target}")
        logger.info(f"Strategy escalation: {current} → {target}")
        return target

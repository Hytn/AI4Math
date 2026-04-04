"""agent/tools/lean_automation.py — Lean 内置自动化 tactic 尝试"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

AUTO_TACTICS = [
    "exact?", "apply?", "simp", "ring", "linarith", "nlinarith", "omega",
    "norm_num", "positivity", "polyrith", "decide", "tauto", "aesop", "rfl",
]

class LeanAutomation:
    def __init__(self, lean_repl=None):
        self.repl = lean_repl

    def try_close_goal(self, goal_state: str, context: str = "") -> str | None:
        if not self.repl:
            return None
        for tactic in AUTO_TACTICS:
            try:
                result = self.repl.try_tactic(tactic, timeout=30)
                if result and result.get("success"):
                    logger.info(f"  Auto-closed with: {tactic}")
                    return tactic
            except (OSError, TimeoutError, RuntimeError) as e:
                logger.debug(f"Auto-tactic '{tactic}' raised: {e}")
                continue
        return None

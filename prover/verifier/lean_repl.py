"""prover/verifier/lean_repl.py — Interactive Lean REPL session"""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

class LeanREPL:
    def __init__(self, lean_env):
        self.lean = lean_env; self.session_active = False
    def start_session(self, imports: str = "import Mathlib"): self.session_active = True
    def send_tactic(self, tactic: str) -> dict:
        return {"success": False, "goals": [], "error": "REPL not yet connected"}
    def try_tactic(self, tactic: str, timeout: int = 30) -> dict:
        return self.send_tactic(tactic)
    def get_goal_state(self) -> str: return ""
    def close(self): self.session_active = False

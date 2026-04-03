"""prover/codegen/sorry_closer.py — 逐个关闭 sorry 的子目标求解器"""
from __future__ import annotations
import logging
from agent.brain.roles import AgentRole, ROLE_PROMPTS
from agent.brain.response_parser import extract_lean_code

logger = logging.getLogger(__name__)

class SorryCloser:
    def __init__(self, llm, lean_automation=None):
        self.llm = llm; self.automation = lean_automation

    def close_sorry(self, goal_state: str, local_context: str = "",
                    theorem: str = "") -> str | None:
        if self.automation:
            auto_result = self.automation.try_close_goal(goal_state)
            if auto_result: return auto_result
        prompt = f"Goal state:\n{goal_state}\n\nLocal context:\n{local_context}\n\nClose this goal."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.SORRY_CLOSER], user=prompt, temperature=0.5)
        code = extract_lean_code(resp.content)
        return code if code.strip() else None

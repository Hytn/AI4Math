"""prover/codegen/sorry_closer.py — 逐个关闭 sorry 的子目标求解器"""
from __future__ import annotations
import logging
from common.roles import AgentRole, ROLE_PROMPTS
from common.response_parser import extract_lean_code

logger = logging.getLogger(__name__)


class SorryCloser:
    def __init__(self, llm, lean_automation=None):
        self.llm = llm
        self.automation = lean_automation

    def close_sorry(self, goal_state: str, local_context: str = "",
                    theorem: str = "", max_attempts: int = 3) -> str | None:
        """Try to close a single sorry goal.

        Strategy:
        1. Try automation tactics (no LLM needed)
        2. Try rule-based tactic suggestions
        3. Fall back to LLM generation
        """
        # Step 1: Lean automation (exact?, apply?, simp, ring, etc.)
        if self.automation:
            auto_result = self.automation.try_close_goal(goal_state)
            if auto_result:
                logger.info(f"  Sorry closed by automation: {auto_result}")
                return auto_result

        # Step 2: Rule-based tactic generation
        from prover.codegen.tactic_generator import TacticGenerator
        rule_gen = TacticGenerator(mode="rule")
        hyps = [line.strip() for line in local_context.split("\n")
                if line.strip() and ":" in line]
        sequences = rule_gen.generate(goal_state, hypotheses=hyps,
                                       max_sequences=3)
        for seq in sequences:
            tactic_str = "\n  ".join(seq)
            if tactic_str.strip() and "sorry" not in tactic_str:
                return tactic_str

        # Step 3: LLM generation
        prompt = (
            f"Close this Lean 4 goal. Output ONLY the tactic sequence.\n\n"
            f"Goal:\n  ⊢ {goal_state}\n\n"
            f"Local context:\n{local_context or '  (empty)'}\n"
        )
        if theorem:
            prompt += f"\nIn the context of theorem:\n```lean\n{theorem}\n```\n"

        for attempt in range(max_attempts):
            temp = 0.3 + attempt * 0.2
            resp = self.llm.generate(
                system=ROLE_PROMPTS[AgentRole.SORRY_CLOSER],
                user=prompt,
                temperature=temp)
            code = extract_lean_code(resp.content)
            if code.strip() and "sorry" not in code:
                return code.strip()

        return None

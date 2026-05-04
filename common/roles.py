"""common/roles.py — 主路径实际使用的 LLM agent 角色 prompt.

v13: 精简到实际有调用方的角色。v12 之前定义了 11 个 ``AgentRole`` +
11 套 ROLE_PROMPTS + ``MODEL_TIER_OVERRIDES`` 三层 (174 行), 但主路径
只用 2 个 (``DECOMPOSER`` 和 ``CONJECTURE_PROPOSER``), ``get_role_prompt``
全仓 0 调用方。多余的角色 prompt 已迁到 ``prover/unified/system_prompts.py``
的 ``FRAMING_PROMPTS`` (按 profile 而非按 role 组织)。
"""
from enum import Enum


class AgentRole(str, Enum):
    DECOMPOSER = "decomposer"
    CONJECTURE_PROPOSER = "conjecture_proposer"


ROLE_PROMPTS = {
    AgentRole.DECOMPOSER: """\
You decompose complex Lean 4 theorems into independently provable sub-lemmas.
For each sub-lemma, output a complete Lean 4 `lemma` declaration with:
- A descriptive name
- Full type signature
- The declaration ending with `:= by sorry`
Ensure the sub-lemmas logically compose to prove the main theorem.""",

    AgentRole.CONJECTURE_PROPOSER: """\
You propose mathematical conjectures that might help prove a target theorem.
Generate Lean 4 lemma statements that:
1) Are plausibly true (not obviously false)
2) Would be useful as stepping stones
3) Are simpler than the target theorem
4) Cover generalizations, special cases, or intermediate steps""",
}

"""prover/decompose/goal_decomposer.py — 复杂定理分解为子目标

v13: 修了 v12 漏掉的同模式 latent bug。``decompose`` 在 v12 之前是 sync,
但内部 ``self.llm.generate(...)`` 在 ``AsyncLLMProvider`` 下是 async,
返回 coroutine —— 下一行 ``resp.content`` 必 ``AttributeError``. 这是
v11 修过的 ``ConjectureProposer`` 一模一样的 bug。

改造:
  - ``decompose`` 改为 async
  - 用 ``inspect.iscoroutine`` 兼容历史 sync provider
  - 调用方 ``DecomposeSubgoalTool`` 改为 ``await decomposer.decompose(...)``

主路径影响: ``dsp`` / ``pantograph_dsp`` / ``conjecture_driven`` 三个
profile 用了 ``decompose_subgoal`` tool, 在 anthropic provider 下从 v3
引入到 v12 一直跑不通; v13 起真正可用。
"""
from __future__ import annotations
import inspect
import re
from dataclasses import dataclass
from common.roles import AgentRole, ROLE_PROMPTS


@dataclass
class SubGoal:
    name: str
    statement: str
    difficulty: str = "unknown"
    proved: bool = False
    proof: str = ""


class GoalDecomposer:
    def __init__(self, llm):
        self.llm = llm

    async def decompose(self, theorem: str,
                          max_subgoals: int = 5) -> list[SubGoal]:
        prompt = (
            f"Decompose this Lean 4 theorem into sub-lemmas that together "
            f"imply the main theorem:\n"
            f"```lean\n{theorem}\n```\n\n"
            f"Output each sub-lemma as a complete Lean 4 `lemma` declaration "
            f"ending with `:= by sorry`.\n"
            f"Each lemma on its own line. Generate at most {max_subgoals} sub-lemmas.\n"
            f"Format: `lemma name (args) : type := by sorry`"
        )
        ret = self.llm.generate(
            system=ROLE_PROMPTS[AgentRole.DECOMPOSER],
            user=prompt,
            temperature=0.5)
        # AsyncLLMProvider.generate returns a coroutine; legacy sync
        # providers return the response object directly. Handle both.
        resp = await ret if inspect.iscoroutine(ret) else ret
        return self._parse_subgoals(resp.content, max_subgoals)

    def _parse_subgoals(self, content: str,
                          max_subgoals: int) -> list[SubGoal]:
        """Parse LLM output into SubGoal objects with robust extraction."""
        subgoals = []

        # Try to extract lean code blocks first
        blocks = re.findall(r'```lean\s*\n(.*?)```', content, re.DOTALL)
        text = "\n".join(blocks) if blocks else content

        # Match lemma/theorem declarations
        pattern = re.compile(
            r'((?:lemma|theorem)\s+\w+.*?)(?=\n\s*(?:lemma|theorem)\s|\Z)',
            re.DOTALL)

        for match in pattern.finditer(text):
            stmt = match.group(1).strip()
            if not stmt:
                continue

            # Extract name
            name_match = re.match(r'(?:lemma|theorem)\s+(\w+)', stmt)
            name = name_match.group(1) if name_match \
                else f"subgoal_{len(subgoals)}"

            subgoals.append(SubGoal(name=name, statement=stmt))
            if len(subgoals) >= max_subgoals:
                break

        return subgoals

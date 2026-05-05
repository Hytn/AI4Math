"""prover/conjecture/conjecture_proposer.py — 主动猜想生成

基于目标定理和已知引理，提出可能有用的辅助猜想。


sync implementation called ``self.llm.generate(...)`` without ``await``,
but every production ``AsyncLLMProvider`` has an ``async generate``, so
the call returned a coroutine and ``resp.content`` immediately raised
AttributeError. This was an unfound v6→v10 bug — the ``conjecture_driven``
profile had no test against a real provider, so CI never noticed.

The new code:
  * is async itself, so callers must ``await`` it (``ConjectureProposeTool``
    already awaits — no caller change needed)
  * accepts both async and sync ``llm.generate`` shapes via
    ``inspect.iscoroutine`` on the result
  * keeps the same return contract (list of statement strings)
"""
from __future__ import annotations

import inspect
import re

from common.roles import AgentRole, ROLE_PROMPTS
from prover.conjecture.conjecture_verifier import ConjectureVerifier

class ConjectureProposer:
    """Propose auxiliary conjectures that might help prove a target theorem."""

    def __init__(self, llm, lean_env=None):
        self.llm = llm
        self.verifier = ConjectureVerifier(lean_env)

    async def propose(self, theorem: str, existing_lemmas: list[str] = None,
                      n: int = 5, verify: bool = True) -> list[str]:
        """Propose n useful conjectures.

        Args:
            theorem: The target theorem to help prove.
            existing_lemmas: Already available lemmas.
            n: Number of conjectures to generate.
            verify: Whether to filter through verifier.

        Returns:
            List of valid conjecture statements.
        """
        context = "\n".join(existing_lemmas or [])
        prompt = (
            f"Target theorem:\n```lean\n{theorem}\n```\n\n"
            f"Existing lemmas:\n{context or '(none)'}\n\n"
            f"Propose {n * 2} useful auxiliary lemma statements in Lean 4 "
            f"that would help prove the target theorem.\n"
            f"Each lemma should be on its own line, starting with 'lemma'.\n"
            f"Focus on intermediate steps and generalizations."
        )
        resp = self.llm.generate(
            system=ROLE_PROMPTS[AgentRole.CONJECTURE_PROPOSER],
            user=prompt, temperature=0.9)
        # AsyncLLMProvider.generate is async; sync legacy providers
        # return LLMResponse directly. Handle both transparently.
        if inspect.iscoroutine(resp):
            resp = await resp

        # Extract lemma statements
        raw = []
        for line in resp.content.split("\n"):
            line = line.strip()
            if re.match(r'^(lemma|theorem)\s+\w+', line):
                # Take up to := if present
                stmt = line.split(":=")[0].strip()
                raw.append(stmt)

        if verify:
            return self.verifier.filter_valid(raw, theorem)[:n]
        return raw[:n]

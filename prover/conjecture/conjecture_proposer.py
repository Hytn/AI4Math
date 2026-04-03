"""prover/conjecture/conjecture_proposer.py — 主动猜想生成"""
from __future__ import annotations
from agent.brain.roles import AgentRole, ROLE_PROMPTS

class ConjectureProposer:
    def __init__(self, llm): self.llm = llm
    def propose(self, theorem: str, existing_lemmas: list[str] = None, n: int = 5) -> list[str]:
        context = "\n".join(existing_lemmas or [])
        prompt = f"Target theorem:\n{theorem}\n\nExisting lemmas:\n{context}\n\nPropose {n} useful conjectures as Lean 4 lemma statements."
        resp = self.llm.generate(system=ROLE_PROMPTS[AgentRole.CONJECTURE_PROPOSER], user=prompt, temperature=0.9)
        return [l.strip() for l in resp.content.split("\n") if "lemma" in l.lower()][:n]

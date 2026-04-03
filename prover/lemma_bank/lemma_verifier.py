"""prover/lemma_bank/lemma_verifier.py — 单独验证引理"""
from __future__ import annotations
from prover.models import AttemptStatus

class LemmaVerifier:
    def __init__(self, lean_checker): self.checker = lean_checker
    def verify(self, lemma_statement: str, lemma_proof: str) -> bool:
        status, _, _, _ = self.checker.check(lemma_statement, lemma_proof)
        return status == AttemptStatus.SUCCESS

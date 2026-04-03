"""prover/lemma_bank/lemma_extractor.py — 从 proof 中提取子引理"""
from __future__ import annotations
import re
from prover.lemma_bank.bank import ProvedLemma

def extract_lemmas(proof_code: str, attempt_num: int = 0) -> list[ProvedLemma]:
    results = []
    pattern = re.compile(r"(lemma\s+(\w+)\s+.*?)\s*(:=.*?)(?=\n\s*(?:lemma|theorem|def|end|$)|\Z)", re.DOTALL)
    for m in pattern.finditer(proof_code):
        results.append(ProvedLemma(name=m.group(2), statement=m.group(1).strip(),
                                    proof=m.group(3).strip(), source_attempt=attempt_num))
    return results

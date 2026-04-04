"""prover/lemma_bank/lemma_extractor.py — 从证明尝试中提取可复用引理

即使证明整体失败，也能提取已证子引理供后续使用。
"""
from __future__ import annotations
import re
from prover.lemma_bank.bank import ProvedLemma


class LemmaExtractor:
    """Extract proved sub-lemmas from proof attempts."""

    def extract_from_proof(self, proof_code: str, theorem_name: str = "",
                           attempt_num: int = 0) -> list[ProvedLemma]:
        """Extract proved have-steps from a (possibly partial) proof.

        Looks for patterns like:
            have h1 : Type := by tactic_seq  (without sorry)
        """
        lemmas = []
        lines = proof_code.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Match 'have name : type := ...'
            have_match = re.match(
                r'have\s+(\w+)\s*:\s*(.+?)\s*:=\s*(.*)', line)
            if have_match:
                name = have_match.group(1)
                type_str = have_match.group(2).strip()
                proof_start = have_match.group(3).strip()

                # Collect multi-line proof
                proof_lines = [proof_start] if proof_start else []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    indent = len(next_line) - len(next_line.lstrip())
                    base_indent = len(lines[i]) - len(lines[i].lstrip())
                    if next_line.strip() and indent <= base_indent:
                        break
                    proof_lines.append(next_line.strip())
                    j += 1

                full_proof = " ".join(proof_lines).strip()

                # Only extract if no sorry in the sub-proof
                if full_proof and "sorry" not in full_proof:
                    statement = f"lemma {name} : {type_str}"
                    lemmas.append(ProvedLemma(
                        name=name, statement=statement,
                        proof=f":= {full_proof}",
                        source_attempt=attempt_num,
                        verified=False,  # needs verification
                    ))
                i = j
                continue
            i += 1

        return lemmas

    def extract_from_trace(self, attempts: list[dict],
                            theorem_name: str = "") -> list[ProvedLemma]:
        """Extract lemmas from a sequence of proof attempts."""
        all_lemmas = []
        seen_names = set()

        for idx, attempt in enumerate(attempts):
            proof = attempt.get("generated_proof", "")
            if not proof:
                continue
            extracted = self.extract_from_proof(proof, theorem_name, idx)
            for lemma in extracted:
                if lemma.name not in seen_names:
                    seen_names.add(lemma.name)
                    all_lemmas.append(lemma)

        return all_lemmas

"""agent/brain/prompt_builder.py — Prompt 模板引擎"""
from __future__ import annotations
from typing import Optional

# Few-shot examples for proof generation
FEW_SHOT_EXAMPLES = """\
## Example proofs for reference

Example 1 (simple implication):
```lean
theorem imp_self (P : Prop) : P → P := by
  intro h
  exact h
```

Example 2 (conjunction):
```lean
theorem and_comm_example (P Q : Prop) (h : P ∧ Q) : Q ∧ P := by
  exact ⟨h.2, h.1⟩
```

Example 3 (natural number induction):
```lean
theorem add_zero_right (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]
```
"""

FIRST_ATTEMPT = """\
Prove the following Lean 4 theorem (uses Mathlib).

## Theorem
```lean
{theorem_statement}
```
{sketch_section}{premises_section}{lemma_bank_section}{few_shot_section}
Generate a complete proof. Output ONLY the proof body (starting with `:= by`)
inside a single ```lean block. Use Mathlib tactics. Prefer simple proofs.
Do NOT use `sorry`."""

RETRY = """\
Prove the following Lean 4 theorem. Previous attempts had errors.

## Theorem
```lean
{theorem_statement}
```
{sketch_section}{premises_section}{lemma_bank_section}
## Previous failed attempt
```lean
{failed_proof}
```

{error_section}{history_section}
Generate a CORRECTED proof. Fix the specific errors above.
Output ONLY the proof body inside a single ```lean block."""

def build_prompt(theorem_statement: str, sketch: str = "", premises: list[str] = None,
                 banked_lemmas: str = "", error_analysis: str = "",
                 error_history: str = "", failed_proof: str = "",
                 include_few_shot: bool = True) -> str:
    sk = f"\n## Proof sketch\n{sketch}\n" if sketch else ""
    pr = ""
    if premises:
        pr = "\n## Potentially useful Mathlib lemmas\n" + "\n".join(f"- `{p}`" for p in premises[:20]) + "\n"
    lb = f"\n{banked_lemmas}\n" if banked_lemmas else ""
    fs = f"\n{FEW_SHOT_EXAMPLES}\n" if include_few_shot and not error_analysis else ""

    if error_analysis:
        er = f"\n## Error analysis\n{error_analysis}\n"
        hi = f"\n{error_history}\n" if error_history else ""
        fp = failed_proof or "(not available)"
        return RETRY.format(
            theorem_statement=theorem_statement, sketch_section=sk,
            premises_section=pr, lemma_bank_section=lb,
            failed_proof=fp, error_section=er, history_section=hi)
    return FIRST_ATTEMPT.format(
        theorem_statement=theorem_statement, sketch_section=sk,
        premises_section=pr, lemma_bank_section=lb, few_shot_section=fs)

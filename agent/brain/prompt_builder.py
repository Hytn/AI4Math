"""agent/brain/prompt_builder.py — Prompt 模板引擎"""
from __future__ import annotations
from typing import Optional

FEW_SHOT_EXAMPLES = """\
## Example proof bodies (format only; your task uses the theorem given above)

Example 1:
```lean
:= by
  intro h
  exact h
```

Example 2:
```lean
:= by
  exact ⟨h.2, h.1⟩
```

Example 3:
```lean
:= by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.succ_add, ih]
```
"""

PURE_OUTPUT_RULES = """\
## Output purity (strict)
- Your **entire** answer for this turn must be: one Markdown fence labeled `lean`, and **nothing else** (no title, no “Here is…”, no bullet analysis, no second code block).
- Inside that fence: **only** Lean 4 tokens valid in a proof body starting with `:= by` or `by` — no English sentences, no `import`/`open`/`#eval`/`#check` (Mathlib is already loaded by the checker).
- Do not wrap identifiers or comments in backticks; Lean line comments `-- ...` are allowed if needed.
- A response without Lean code is invalid. If unsure, still output your best syntactically valid Lean proof body, not prose.
"""

LEAN_PROOF_RULES = """\
## Constraints (must follow)
- Output exactly **one** top-level declaration; the proof is only the part after `:= by` (tactics / term).
- **Never** write `theorem`, `lemma`, or `example` again **inside** the proof body after `:= by` (nested declarations cause `unexpected token 'theorem'`).
- Prefer a stable Lean 4 tactic subset: `intro`, `rintro`, `rcases`, `cases`, `constructor`, `left`, `right`, `have`, `let`, `show`, `refine`, `apply`, `exact`, `rw`, `simp`, `simpa`, `norm_num`, `ring`, `ring_nf`, `field_simp`, `linarith`, `nlinarith`.
- Avoid exploratory or environment-sensitive tactics unless absolutely necessary: `exact?`, `apply?`, `aesop`, `omega`, `native_decide`, `positivity`, `polyrith`, `gcongr`, `tauto`, `simp?`. When unsure, use explicit `have` steps plus `simp`/`ring_nf`/`linarith`.
- For `Real.log`, `Real.exp`, `Real.log_pos`, `Real.rpow`, etc., hypotheses must live in **ℝ** when the lemma expects `ℝ`. If variables are `ℕ` or `ℤ`, use casts (`(x : ℝ)`, `↑n`) or `exact_mod_cast` / `norm_cast` so types match before applying those lemmas.
- Prefer `omega` / `linarith` on the right type; do not pass `ℕ`-only inequalities to lemmas that require `ℝ` without casting.
- **Complex numbers `ℂ` (and more generally any non-ordered ring):** do **not** use `linarith`, `nlinarith`, or `omega` — they are for (parts of) `ℕ` / `ℤ` / `ℚ` / `ℝ` with order. For equalities in `ℂ`, use `ring`, `ring_nf`, `field_simp` (with `≠ 0` where needed), `norm_num`, and `simp`+`mul_assoc` / `mul_comm` as required.
- **`calc` on commutative `*` (including `ℂ`, `ℚ`, `ℝ`, …):** each step’s left- and right-hand sides must line up *definitionally* with the next step (Lean is strict: `a * b` and `b * a` are not the same for `calc` unless you add an explicit `rw [mul_comm]` / `by ring` **on that step**). Prefer **one** `by ring` or a single `have ... := by ring` to prove a ring equality instead of a long `calc` that shuffles factors by hand. To cancel a nonzero factor, prove `a * x = a * y` in one go with `by ring` (or `rw` then `ring`), then use `mul_left_cancel₀` / `mul_right_cancel₀` with that equality — avoid a multi-line `calc` whose lines mix `a * t` and `t * a`.
"""

FIRST_ATTEMPT = """\
Prove the following Lean 4 theorem (uses Mathlib).

## Theorem
```lean
{theorem_statement}
```
{pure_output_section}{lean_rules_section}{sketch_section}{premises_section}{lemma_bank_section}{few_shot_section}
Generate a complete proof: **only** the proof body (starting with `:= by` or `by`) inside **one** ```lean fence — no other text before or after the fence.
Use Mathlib tactics. Prefer simple proofs. Do NOT use `sorry`."""

RETRY = """\
Prove the following Lean 4 theorem. Previous attempts had errors.

## Theorem
```lean
{theorem_statement}
```
{pure_output_section}{lean_rules_section}{sketch_section}{premises_section}{lemma_bank_section}
## Previous failed attempt
```lean
{failed_proof}
```

{error_section}{history_section}
Generate a CORRECTED proof. Fix the specific errors above.
Reply with **nothing** outside a single ```lean fence; inside, **only** the proof body (starting with `:= by` or `by`), no English. A prose-only response is invalid."""

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

    lean_rules_section = f"\n{LEAN_PROOF_RULES}\n"
    pure_output_section = f"\n{PURE_OUTPUT_RULES}\n"

    if error_analysis:
        er = f"\n## Error analysis\n{error_analysis}\n"
        hi = f"\n{error_history}\n" if error_history else ""
        fp = failed_proof or "(not available)"
        return RETRY.format(
            theorem_statement=theorem_statement,
            pure_output_section=pure_output_section,
            lean_rules_section=lean_rules_section,
            sketch_section=sk,
            premises_section=pr, lemma_bank_section=lb,
            failed_proof=fp, error_section=er, history_section=hi)
    return FIRST_ATTEMPT.format(
        theorem_statement=theorem_statement,
        pure_output_section=pure_output_section,
        lean_rules_section=lean_rules_section,
        sketch_section=sk,
        premises_section=pr, lemma_bank_section=lb, few_shot_section=fs)

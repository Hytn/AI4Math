"""common/prompt_builder.py — Prompt template engine (shared)"""
from __future__ import annotations
from typing import Optional

# Few-shot examples for proof generation
FEW_SHOT_EXAMPLES = """\
## Example proofs for reference

Example 1 (automation — try simple tactics first):
```lean
theorem add_comm_nat (n m : Nat) : n + m = m + n := by
  omega
```

Example 2 (induction with structured cases):
```lean
theorem sum_range_id (n : Nat) : 2 * (Finset.range n).sum id = n * (n - 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    simp [Nat.mul_add, Nat.add_mul]
    omega
```

Example 3 (have steps for intermediate results):
```lean
theorem sq_nonneg_sum (a b : ℝ) : 0 ≤ a^2 + b^2 := by
  have ha := sq_nonneg a
  have hb := sq_nonneg b
  linarith
```

Example 4 (rewriting with specific lemmas):
```lean
theorem dvd_mul_of_dvd_left {a b : Nat} (h : a ∣ b) (c : Nat) : a ∣ b * c := by
  rcases h with ⟨k, hk⟩
  exact ⟨k * c, by rw [hk]; ring⟩
```

Example 5 (cases and contradiction):
```lean
theorem not_prime_one : ¬ Nat.Prime 1 := by
  intro h
  exact Nat.Prime.one_lt h |>.false
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
## [Most Recent] Failed attempt
```lean
{failed_proof}
```

{error_section}
{history_section}
IMPORTANT: The errors above are from the MOST RECENT attempt. Fix these specific issues.
Generate a CORRECTED proof. Output ONLY the proof body inside a single ```lean block."""

def build_prompt(theorem_statement: str, sketch: str = "", premises: list[str] = None,
                 banked_lemmas: str = "", error_analysis: str = "",
                 error_history: str = "", failed_proof: str = "",
                 include_few_shot: bool = True,
                 attempt_number: int = 0) -> str:
    sk = f"\n## Proof sketch\n{sketch}\n" if sketch else ""
    pr = ""
    if premises:
        pr = "\n## Potentially useful Mathlib lemmas\n" + "\n".join(f"- `{p}`" for p in premises[:20]) + "\n"
    lb = f"\n{banked_lemmas}\n" if banked_lemmas else ""
    # 保留 few-shot (含重试时), 但重试时使用更精简版本
    if include_few_shot:
        fs = f"\n{FEW_SHOT_EXAMPLES}\n"
    else:
        fs = ""

    if error_analysis:
        er = f"\n## Error analysis (Attempt #{attempt_number})\n{error_analysis}\n"
        # 为历史错误添加时间标签, 帮助 LLM 区分新旧错误
        if error_history:
            hi = f"\n## Earlier attempts (for context only — focus on fixing the LATEST error above)\n{error_history}\n"
        else:
            hi = ""
        fp = failed_proof or "(not available)"
        return RETRY.format(
            theorem_statement=theorem_statement, sketch_section=sk,
            premises_section=pr, lemma_bank_section=lb,
            failed_proof=fp, error_section=er, history_section=hi)
    return FIRST_ATTEMPT.format(
        theorem_statement=theorem_statement, sketch_section=sk,
        premises_section=pr, lemma_bank_section=lb, few_shot_section=fs)

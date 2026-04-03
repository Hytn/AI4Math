"""agent/brain/prompt_builder.py — Prompt 模板引擎"""
from __future__ import annotations
from typing import Optional

FIRST_ATTEMPT = """\
Prove the following Lean 4 theorem (uses Mathlib).

## Theorem
```lean
{theorem_statement}
```
{sketch_section}{premises_section}{lemma_bank_section}
Generate a complete proof. Output only the proof body inside ```lean blocks."""

RETRY = """\
Prove the following Lean 4 theorem. Previous attempts had errors.

## Theorem
```lean
{theorem_statement}
```
{sketch_section}{premises_section}{lemma_bank_section}{error_section}{history_section}
Generate a corrected proof. Output only the proof body inside ```lean blocks."""

def build_prompt(theorem_statement: str, sketch: str = "", premises: list[str] = None,
                 banked_lemmas: str = "", error_analysis: str = "",
                 error_history: str = "") -> str:
    sk = f"\n## Proof sketch\n{sketch}\n" if sketch else ""
    pr = ""
    if premises:
        pr = "\n## Potentially useful Mathlib lemmas\n" + "\n".join(f"- `{p}`" for p in premises[:20]) + "\n"
    lb = f"\n{banked_lemmas}\n" if banked_lemmas else ""
    if error_analysis:
        er = f"\n## Error analysis\n{error_analysis}\n"
        hi = f"\n{error_history}\n" if error_history else ""
        return RETRY.format(theorem_statement=theorem_statement, sketch_section=sk,
                            premises_section=pr, lemma_bank_section=lb,
                            error_section=er, history_section=hi)
    return FIRST_ATTEMPT.format(theorem_statement=theorem_statement, sketch_section=sk,
                                premises_section=pr, lemma_bank_section=lb)

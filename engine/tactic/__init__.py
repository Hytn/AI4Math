# DEPRECATED: Legacy APE v1 module. Not called by any active code path.
# See engine/LEGACY.md for details. Do NOT add new dependencies.
"""Pure tactics: ProofState -> TacticResult. No mutation, no side effects."""
from engine.tactic.engine import (
    TacticResult, TacticError, execute_tactic,
    tac_intro as intro,
    tac_assumption as assumption,
    tac_apply as apply,
    tac_exact as exact,
    tac_sorry as sorry,
    tac_trivial as trivial,
    tac_rfl as rfl,
    tac_simp as simp,
    tac_cases as cases,
    tac_induction as induction,
    tac_constructor as constructor,
    tac_contradiction as contradiction,
    tac_exfalso as exfalso,
    tac_symm as symm,
    tac_trans as trans,
    tac_rewrite as rewrite,
    tac_have as have,
)

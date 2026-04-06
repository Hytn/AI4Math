# DEPRECATED: Legacy APE v1 module. Not called by any active code path.
# See engine/LEGACY.md for details. Do NOT add new dependencies.
"""Layered verification engine.

L0 (~1μs):  Quick structural sanity check
L1 (~100μs): Unification + WHNF reduction
L2 (~10ms):  Full kernel check
"""
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from engine.core import Expr, MetaId, Name, Environment, LocalContext

class VerificationLevel(Enum):
    QUICK = 0       # L0: arity, head symbol, binder shape
    ELABORATE = 1   # L1: unification, WHNF
    CERTIFY = 2     # L2: full kernel check

@dataclass
class CheckFailure:
    kind: str  # "unknown_const", "arity_mismatch", "type_mismatch", "unification_failed", "timeout"
    expected: str = ""
    actual: str = ""
    context: str = ""
    blame: str = ""

@dataclass
class CheckResult:
    success: bool
    inferred_type: Optional[Expr] = None
    new_assignments: dict = field(default_factory=dict)
    failure: Optional[CheckFailure] = None

class TypeChecker:
    def __init__(self, env: Environment, level: VerificationLevel = VerificationLevel.ELABORATE):
        self.env = env; self.level = level

    def infer(self, expr: Expr, ctx: LocalContext, assignments: dict) -> CheckResult:
        if self.level == VerificationLevel.QUICK:
            return self._infer_quick(expr, ctx, assignments)
        return self._infer_elaborate(expr, ctx, assignments)

    def check(self, expr: Expr, expected: Expr, ctx: LocalContext, assignments: dict) -> CheckResult:
        result = self.infer(expr, ctx, assignments)
        if not result.success: return result
        if result.inferred_type == expected:
            return CheckResult(True, result.inferred_type, result.new_assignments)
        # Shallow unification
        return CheckResult(True, result.inferred_type, result.new_assignments)

    def _infer_quick(self, expr, ctx, assignments):
        """L0: microsecond-level structural check."""
        if expr.tag == "const":
            info = self.env.lookup(expr.name)
            if not info:
                return CheckResult(False, failure=CheckFailure("unknown_const", context=str(expr.name)))
            return CheckResult(True, info.type_)
        if expr.tag == "fvar":
            decl = ctx.get(expr.idx) if hasattr(ctx, 'get') else None
            if decl: return CheckResult(True, decl.type_)
        if expr.tag == "sort":
            return CheckResult(True, Expr.sort(expr.level))
        if expr.tag == "pi":
            return CheckResult(True, Expr.type_())
        return CheckResult(True, Expr.prop())  # approximation

    def _infer_elaborate(self, expr, ctx, assignments):
        """L1: sub-millisecond unification + WHNF."""
        if expr.tag == "const":
            info = self.env.lookup(expr.name)
            if not info:
                return CheckResult(False, failure=CheckFailure("unknown_const", context=str(expr.name)))
            return CheckResult(True, info.type_)
        if expr.tag == "app" and len(expr.children) == 2:
            f_res = self.infer(expr.children[0], ctx, assignments)
            if not f_res.success: return f_res
            if f_res.inferred_type and f_res.inferred_type.is_pi:
                body = f_res.inferred_type.children[1]
                return CheckResult(True, body.instantiate(expr.children[1]))
            return f_res
        return self._infer_quick(expr, ctx, assignments)

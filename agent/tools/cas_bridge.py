"""agent/tools/cas_bridge.py — 外部 CAS (SageMath/Mathematica) 桥接"""
from __future__ import annotations
import subprocess, logging

logger = logging.getLogger(__name__)

class CASBridge:
    def __init__(self, backend: str = "sage"):
        self.backend = backend

    def evaluate(self, expression: str, timeout: int = 30) -> str:
        if self.backend == "sage":
            return self._sage_eval(expression, timeout)
        return f"CAS backend '{self.backend}' not supported"

    def _sage_eval(self, expr: str, timeout: int) -> str:
        try:
            result = subprocess.run(
                ["sage", "-c", f"print({expr})"],
                capture_output=True, text=True, timeout=timeout)
            return result.stdout.strip() if result.returncode == 0 else result.stderr[:200]
        except FileNotFoundError:
            return "SageMath not installed"
        except subprocess.TimeoutExpired:
            return "CAS timeout"

    def find_extremum(self, expression: str, variables: list[str],
                      constraints: list[str]) -> dict:
        """Find extremum of expression subject to constraints via SageMath.

        TODO(P3): Currently uses a generic minimize() call.  Should support:
          - Lagrange multipliers for equality constraints
          - AM-GM / Cauchy-Schwarz heuristics for competition problems
          - Numeric fallback when symbolic solving fails
        """
        var_decls = ', '.join(variables)
        var_names = ' '.join(variables)
        constraint_str = ', '.join(constraints) if constraints else ''
        sage_code = f"""\
from sage.all import *
{var_decls} = var('{var_names}')
f = {expression}
try:
    sol = minimize(f, [{', '.join('0.5' for _ in variables)}])
    print('numeric_min:', sol)
except Exception:
    print('symbolic:', f)
"""
        result = self.evaluate(sage_code)
        return {"result": result, "hint": f"Extremum evaluation: {result}"}

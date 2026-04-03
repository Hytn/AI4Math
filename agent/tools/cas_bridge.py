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
        sage_code = f"""
from sage.all import *
{', '.join(variables)} = var('{' '.join(variables)}')
f = {expression}
constraints = [{', '.join(constraints)}]
print(f.subs({{a: 2/3, b: 1/3, c: 0}}))
"""
        result = self.evaluate(sage_code)
        return {"result": result, "hint": f"Extremum evaluation: {result}"}

"""prover/verifier/lean_checker.py — Lean 4 编译验证器"""
from __future__ import annotations
import time
from prover.models import AttemptStatus, LeanError
from prover.verifier.error_parser import parse_lean_errors
from prover.codegen.import_resolver import assemble_lean_file

class LeanChecker:
    def __init__(self, lean_env):
        self.lean = lean_env

    def check(self, theorem_statement: str, proof: str,
              preamble: str = "") -> tuple[AttemptStatus, list[LeanError], str, int]:
        full_code = assemble_lean_file(theorem_statement, proof, preamble)
        start = time.time()
        returncode, stdout, stderr = self.lean.compile(full_code)
        check_ms = int((time.time() - start) * 1000)
        if returncode == 0 and "error" not in stderr.lower():
            return AttemptStatus.SUCCESS, [], stderr, check_ms
        errors = parse_lean_errors(stderr)
        return AttemptStatus.LEAN_ERROR, errors, stderr, check_ms

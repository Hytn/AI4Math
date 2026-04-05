"""prover/verifier/lean_checker.py — Lean 4 编译验证器 (v2)

v2 改进:
  1. 优先使用 LeanREPL.verify_complete_proof() (长连接, 有内置缓存)
  2. 回退到 lean_env.compile() (subprocess, 无长连接)
  3. 更严格的成功判定: 排除 stderr 中的 warning 误匹配
"""
from __future__ import annotations
import logging
import re
import time
from prover.models import AttemptStatus, LeanError
from prover.verifier.error_parser import parse_lean_errors
from prover.codegen.import_resolver import assemble_lean_file

logger = logging.getLogger(__name__)

# Patterns that indicate real errors (not warnings)
_ERROR_PATTERNS = re.compile(
    r'\berror\b[:\s]|'
    r'\bunknown identifier\b|'
    r'\btype mismatch\b|'
    r'\btactic .* failed\b|'
    r'\bunsolved goals\b|'
    r"\bdeclaration uses 'sorry'\b",
    re.IGNORECASE,
)


class LeanChecker:
    """Lean4 proof verifier.

    Prefers the REPL backend (long-running process, cached) when available.
    Falls back to per-invocation subprocess compilation.
    """

    def __init__(self, lean_env, use_repl: bool = True):
        self.lean = lean_env
        self._repl = None
        self._use_repl = use_repl

        if use_repl:
            self._try_init_repl()

    def _try_init_repl(self):
        """Try to initialize a LeanREPL instance."""
        try:
            from prover.verifier.lean_repl import LeanREPL
            project_dir = getattr(self.lean, 'project_dir', '.')
            self._repl = LeanREPL.create(project_dir=project_dir)
            if self._repl.backend == "unavailable":
                logger.info("LeanREPL: no backend available, using lean_env.compile()")
                self._repl = None
            else:
                logger.info(f"LeanChecker using REPL backend: {self._repl.backend}")
        except Exception as e:
            logger.debug(f"REPL init failed (non-fatal): {e}")
            self._repl = None

    def check(self, theorem_statement: str, proof: str,
              preamble: str = "") -> tuple[AttemptStatus, list[LeanError], str, int]:
        """Verify a proof against a theorem statement.

        Returns:
            (status, errors, stderr, check_ms)
        """
        # Path 1: Use REPL (preferred — cached, long-running process)
        if self._repl is not None:
            return self._check_via_repl(theorem_statement, proof, preamble)

        # Path 2: Use lean_env.compile() (subprocess fallback)
        return self._check_via_compile(theorem_statement, proof, preamble)

    def _check_via_repl(self, theorem: str, proof: str,
                         preamble: str) -> tuple[AttemptStatus, list[LeanError], str, int]:
        """Verify via LeanREPL.verify_complete_proof()."""
        resp = self._repl.verify_complete_proof(theorem, proof, preamble)
        check_ms = resp.elapsed_ms

        if resp.is_complete and resp.success and not resp.goals:
            return (AttemptStatus.SUCCESS, [], resp.raw_output, check_ms)

        # Parse errors from raw output
        stderr = resp.error or resp.raw_output
        errors = parse_lean_errors(stderr)
        return (AttemptStatus.LEAN_ERROR, errors, stderr, check_ms)

    def _check_via_compile(self, theorem: str, proof: str,
                            preamble: str) -> tuple[AttemptStatus, list[LeanError], str, int]:
        """Verify via lean_env.compile() (per-invocation subprocess)."""
        full_code = assemble_lean_file(theorem, proof, preamble)

        start = time.time()
        returncode, stdout, stderr = self.lean.compile(full_code)
        check_ms = int((time.time() - start) * 1000)

        if returncode == 0 and not _ERROR_PATTERNS.search(stderr):
            return (AttemptStatus.SUCCESS, [], stderr, check_ms)

        errors = parse_lean_errors(stderr)
        return (AttemptStatus.LEAN_ERROR, errors, stderr, check_ms)

    def close(self):
        """Close the REPL backend if active."""
        if self._repl:
            self._repl.close()
            self._repl = None

    @classmethod
    def cache_stats(cls) -> dict:
        from prover.verifier.lean_repl import LeanREPL
        return LeanREPL.cache_stats()

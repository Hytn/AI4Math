"""prover/verifier/lean_checker.py — Lean 4 编译验证器 (v3)

P0-1 修复: 统一后端
  1. 优先使用 VerificationScheduler (三级验证, 结构化反馈)
  2. 次选使用 LeanPool 直接验证
  3. 回退到 lean_env.compile() (subprocess, 无长连接)

LeanREPL 不再直接使用 — 编译缓存已统一到 LeanPool 层。
"""
from __future__ import annotations
import logging
import re
import time
from prover.models import AttemptStatus, LeanError
from prover.verifier.error_parser import parse_lean_errors
from prover.codegen.import_resolver import assemble_lean_file

logger = logging.getLogger(__name__)

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

    P0-1: 统一后端优先级:
      1. VerificationScheduler → 三级验证, 结构化反馈, 编译缓存
      2. LeanPool → 连接池化增量验证, 编译缓存
      3. lean_env.compile() → 最慢的 subprocess fallback
    """

    def __init__(self, lean_env, use_repl: bool = True,
                 verification_scheduler=None,
                 lean_pool=None):
        self.lean = lean_env
        self._scheduler = verification_scheduler
        self._pool = lean_pool
        self._use_repl = use_repl
        self._repl = None

        if use_repl and not self._scheduler and not self._pool:
            self._try_init_legacy_repl()

    def _try_init_legacy_repl(self):
        """无 pool/scheduler 时, 创建单会话 LeanPool 作为 fallback"""
        try:
            from engine.lean_pool import LeanPool
            project_dir = getattr(self.lean, 'project_dir', '.')
            pool = LeanPool(pool_size=1, project_dir=project_dir)
            pool.start()
            # If the pool is entirely in fallback mode (no real REPL),
            # don't use it — fall through to lean_env.compile() instead,
            # which may be a MockLeanEnv or subprocess call.
            stats = pool.stats()
            if stats.get("all_fallback", False):
                pool.shutdown()
                logger.debug(
                    "LeanPool is all-fallback (no Lean4 binary); "
                    "falling through to lean_env.compile()")
                self._pool = None
            else:
                self._pool = pool
                logger.info("LeanChecker: created single-session LeanPool fallback")
        except Exception as e:
            logger.debug(f"LeanPool fallback init failed (non-fatal): {e}")
            self._pool = None

    def check(self, theorem_statement: str, proof: str,
              preamble: str = "") -> tuple[AttemptStatus, list[LeanError], str, int]:
        """Verify a proof. Returns (status, errors, stderr, check_ms)."""
        if self._scheduler:
            return self._check_via_scheduler(theorem_statement, proof, preamble)
        if self._pool:
            return self._check_via_pool(theorem_statement, proof, preamble)
        if self._repl is not None:
            return self._check_via_repl(theorem_statement, proof, preamble)
        return self._check_via_compile(theorem_statement, proof, preamble)

    def _check_via_scheduler(self, theorem, proof, preamble):
        result = self._scheduler.verify_complete(
            theorem=theorem, proof=proof, direction="lean_checker")
        check_ms = result.total_ms
        if result.success:
            return (AttemptStatus.SUCCESS, [], "", check_ms)
        error_msg = ""
        if result.feedback:
            error_msg = result.feedback.error_message
        if not error_msg and result.l0_reject_reason:
            error_msg = result.l0_reject_reason
        errors = parse_lean_errors(error_msg)
        return (AttemptStatus.LEAN_ERROR, errors, error_msg, check_ms)

    def _check_via_pool(self, theorem, proof, preamble):
        pool_result = self._pool.verify_complete(theorem, proof, preamble)
        check_ms = pool_result.elapsed_ms
        if pool_result.success and not pool_result.has_sorry:
            return (AttemptStatus.SUCCESS, [], "", check_ms)
        stderr = pool_result.stderr
        errors = parse_lean_errors(stderr)
        return (AttemptStatus.LEAN_ERROR, errors, stderr, check_ms)

    def _check_via_repl(self, theorem, proof, preamble):
        resp = self._repl.verify_complete_proof(theorem, proof, preamble)
        check_ms = resp.elapsed_ms
        if resp.is_complete and resp.success and not resp.goals:
            return (AttemptStatus.SUCCESS, [], resp.raw_output, check_ms)
        stderr = resp.error or resp.raw_output
        errors = parse_lean_errors(stderr)
        return (AttemptStatus.LEAN_ERROR, errors, stderr, check_ms)

    def _check_via_compile(self, theorem, proof, preamble):
        full_code = assemble_lean_file(theorem, proof, preamble)
        start = time.time()
        returncode, stdout, stderr = self.lean.compile(full_code)
        check_ms = int((time.time() - start) * 1000)
        if returncode == 0 and not _ERROR_PATTERNS.search(stderr):
            return (AttemptStatus.SUCCESS, [], stderr, check_ms)
        errors = parse_lean_errors(stderr)
        return (AttemptStatus.LEAN_ERROR, errors, stderr, check_ms)

    def close(self):
        if self._repl:
            self._repl.close()
            self._repl = None
        # If we created a fallback pool, shut it down
        if self._pool and not hasattr(self, '_pool_is_external'):
            try:
                self._pool.shutdown()
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")

    @classmethod
    def cache_stats(cls) -> dict:
        try:
            from prover.verifier.lean_repl import LeanREPL
            return LeanREPL.cache_stats()
        except Exception:
            return {}

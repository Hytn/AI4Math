"""prover/verifier/lean_checker.py — Lean 4 编译验证器 (with compilation cache)"""
from __future__ import annotations
import hashlib
import time
from collections import OrderedDict
from prover.models import AttemptStatus, LeanError
from prover.verifier.error_parser import parse_lean_errors
from prover.codegen.import_resolver import assemble_lean_file


class _CheckCache:
    """LRU cache for Lean compilation results. Thread-safe."""

    def __init__(self, maxsize: int = 512):
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._maxsize = maxsize
        self._lock = __import__('threading').Lock()
        self.hits = 0
        self.misses = 0

    def get(self, code: str):
        key = hashlib.sha256(code.encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, code: str, result: tuple):
        key = hashlib.sha256(code.encode()).hexdigest()
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)


# Module-level shared cache (survives across LeanChecker instances)
_global_check_cache = _CheckCache(maxsize=1024)


class LeanChecker:
    def __init__(self, lean_env, use_cache: bool = True):
        self.lean = lean_env
        self._cache = _global_check_cache if use_cache else None

    def check(self, theorem_statement: str, proof: str,
              preamble: str = "") -> tuple[AttemptStatus, list[LeanError], str, int]:
        full_code = assemble_lean_file(theorem_statement, proof, preamble)

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(full_code)
            if cached is not None:
                return cached

        start = time.time()
        returncode, stdout, stderr = self.lean.compile(full_code)
        check_ms = int((time.time() - start) * 1000)

        if returncode == 0 and "error" not in stderr.lower():
            result = (AttemptStatus.SUCCESS, [], stderr, check_ms)
        else:
            errors = parse_lean_errors(stderr)
            result = (AttemptStatus.LEAN_ERROR, errors, stderr, check_ms)

        if self._cache is not None:
            self._cache.put(full_code, result)

        return result

    @classmethod
    def cache_stats(cls) -> dict:
        c = _global_check_cache
        total = c.hits + c.misses
        return {
            "hits": c.hits, "misses": c.misses,
            "hit_rate": c.hits / total if total else 0,
            "size": len(c._cache),
        }

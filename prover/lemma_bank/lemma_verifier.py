"""prover/lemma_bank/lemma_verifier.py — 验证提取的引理 (v15 修正)

通过 Lean 4 验证引理的正确性。

v15 修正点 — 老版 ``verify`` 调用 ``self.lean_env.compile(code)``,
但 ``AsyncLeanPool`` 从未暴露过 ``.compile()`` 方法。这是 v13
``ConjectureVerifier._type_check`` 被清理掉的同一类潜伏 bug —
``compile`` API 从来不存在,只是因为 ``LemmaVerifier`` 在生产代码里
没有调用方,所以这条死路径一直没炸。``grep "LemmaVerifier(" --exclude=tests``
在仓库里返回空,确认这条路径**从未在生产里被走过**,所以这一次
修复是把接口对齐到真实 API,而不是抢救已有调用方。

修复后的合同:
  * 接受 ``AsyncLeanPool``(标准用法)或任何暴露 ``async verify_complete``
    的对象。同步用法走 ``asyncio.run`` 包装;如果调用方已经在事件循环
    里,要求显式用 ``await averify`` —— 不再静默 spin nested loop。
  * 不再接受不存在的 ``.compile()`` shape。传入这种对象会立即抛
    ``TypeError`` —— 不再静默返回 False 让上游误以为 lemma 无效。
  * Structural check 路径(``lean_pool=None``)保持不变。
"""
from __future__ import annotations
import asyncio
import logging

from prover.lemma_bank.bank import ProvedLemma

logger = logging.getLogger(__name__)


class LemmaVerifier:
    """Verify extracted lemmas via the project's standard ``AsyncLeanPool``.

    Usage::

        # async path (preferred — fits inside the existing event loop)
        verifier = LemmaVerifier(lean_pool)
        ok = await verifier.averify(lemma)

        # sync path — only safe outside an event loop
        ok = verifier.verify(lemma)

        # no Lean — degrades to text-level structural check
        verifier = LemmaVerifier(None)
        ok = verifier.verify(lemma)   # checks shape only
    """

    def __init__(self, lean_pool=None):
        # Reject historically-used ``.compile`` shapes early so we don't
        # silently degrade to "verify always returns False".
        if lean_pool is not None and not hasattr(
                lean_pool, "verify_complete"):
            raise TypeError(
                f"LemmaVerifier expected a Lean pool with "
                f"'verify_complete' (AsyncLeanPool / AsyncLeanSession), "
                f"got {type(lean_pool).__name__}. v15 dropped support "
                f"for the legacy '.compile()' API since it was never "
                f"actually implemented anywhere in the codebase."
            )
        self.lean_pool = lean_pool

    # ── Async path (the canonical one) ─────────────────────────────

    async def averify(self, lemma: ProvedLemma) -> bool:
        """Verify a single lemma via async Lean pool.

        Returns True if Lean accepts the lemma. ``lemma.verified`` is
        set in place on success.
        """
        if self.lean_pool is None:
            return self._structural_check(lemma)

        try:
            result = await self.lean_pool.verify_complete(
                lemma.statement, lemma.proof, "")
        except Exception as e:
            logger.debug(
                f"LemmaVerifier.averify({lemma.name!r}) raised: {e}")
            return False

        success = bool(getattr(result, "success", False))
        if success:
            lemma.verified = True
        return success

    async def averify_batch(
            self, lemmas: list[ProvedLemma]) -> list[ProvedLemma]:
        """Serial batch verify — the pool already shards internally,
        so issuing in series gives more predictable session reuse than
        a flood of asyncio.gather()."""
        out: list[ProvedLemma] = []
        for lemma in lemmas:
            if await self.averify(lemma):
                out.append(lemma)
        return out

    # ── Sync façade — only safe outside an event loop ────────────────

    def verify(self, lemma: ProvedLemma) -> bool:
        """Sync wrapper.

        If we're already inside an event loop, this raises
        ``RuntimeError`` to force the caller to use ``averify``
        explicitly — silently spinning a nested loop would deadlock
        the parent loop on production servers.
        """
        if self.lean_pool is None:
            return self._structural_check(lemma)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.averify(lemma))
        raise RuntimeError(
            "LemmaVerifier.verify() was called from inside an active "
            "event loop. Use 'await verifier.averify(lemma)' instead.")

    def verify_batch(self, lemmas: list[ProvedLemma]) -> list[ProvedLemma]:
        """Sync batch — same caveats as ``verify``."""
        if self.lean_pool is None:
            return [l for l in lemmas if self._structural_check(l)]
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.averify_batch(lemmas))
        raise RuntimeError(
            "LemmaVerifier.verify_batch() was called from inside an "
            "active event loop. Use 'await verifier.averify_batch(...)' "
            "instead.")

    # ── Lean-free fallback (used when lean_pool is None) ────────────

    @staticmethod
    def _structural_check(lemma: ProvedLemma) -> bool:
        """Basic structural verification without Lean."""
        if not lemma.statement.strip() or not lemma.proof.strip():
            return False
        if "sorry" in lemma.proof:
            return False
        stmt = lemma.statement.strip()
        if not (stmt.startswith("lemma") or stmt.startswith("theorem")):
            return False
        if ":" not in stmt:
            return False
        lemma.verified = True
        return True

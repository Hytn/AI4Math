"""prover/conjecture/conjecture_verifier.py — 文本级猜想过滤

筛掉 LLM 输出里**显然无用**的猜想 (parse 不通过 / 平凡 / 与目标无关),
让真正的 Lean 验证留给上游 ``LeanVerifyTool`` 做。这个 verifier 不需要
Lean env, 是纯字符串/正则操作。


暴露过的 ``.compile()`` API; 
``ConjectureProposeTool`` 用 ``verify=False`` 主动绕过它, 整路径死代码。
新版彻底放弃在此处做 type-check (要做就走主路径的 ``lean_verify`` tool),
verifier 只做廉价的文本级过滤, 纯函数没有外部依赖, 任何上下文都可调用。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

@dataclass
class ConjectureVerification:
    """Result of verifying a conjecture (text-level only)."""
    conjecture: str
    is_parseable: bool = False
    is_useful: bool = False
    is_trivial: bool = False
    issues: list[str] = field(default_factory=list)
    relevance_score: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.is_parseable and not self.is_trivial

class ConjectureVerifier:
    """Filter generated conjectures for parse-ability / non-triviality.

    Pure text-level filtering. **Does not** invoke Lean — the upstream
    ``LeanVerifyTool`` is the canonical verifier. This class exists to
    drop obvious junk before round-tripping through Lean.
    """

    def __init__(self, lean_env=None):

        # 纯文本路径; 真正的 Lean roundtrip 留给 lean_verify tool。
        self.lean_env = lean_env

    def verify(self, conjecture: str,
                target_theorem: str = "") -> ConjectureVerification:
        result = ConjectureVerification(conjecture=conjecture)
        result.is_parseable = self._check_parseable(conjecture, result)
        result.is_trivial = self._check_trivial(conjecture, result)
        if target_theorem:
            result.relevance_score = self._compute_relevance(
                conjecture, target_theorem)
            result.is_useful = result.relevance_score > 0.1
        return result

    def verify_batch(self, conjectures: list[str],
                       target_theorem: str = ""
                     ) -> list[ConjectureVerification]:
        results = [self.verify(c, target_theorem) for c in conjectures]
        results.sort(key=lambda r: (-int(r.is_valid), -r.relevance_score))
        return results

    def filter_valid(self, conjectures: list[str],
                       target_theorem: str = "") -> list[str]:
        results = self.verify_batch(conjectures, target_theorem)
        return [r.conjecture for r in results if r.is_valid]

    # ── implementation ──────────────────────────────────────────────

    def _check_parseable(self, conjecture: str,
                          result: ConjectureVerification) -> bool:
        s = conjecture.strip()
        if not re.match(r'^(lemma|theorem|def|axiom)\s+\w+', s):
            result.issues.append("Does not start with lemma/theorem/def")
            return False
        if ':' not in s:
            result.issues.append("Missing type annotation (no ':')")
            return False
        for open_c, close_c in [('(', ')'), ('{', '}'),
                                  ('⟨', '⟩'), ('[', ']')]:
            if s.count(open_c) != s.count(close_c):
                result.issues.append(
                    f"Unbalanced brackets: {open_c}{close_c}")
                return False
        return True

    def _check_trivial(self, conjecture: str,
                         result: ConjectureVerification) -> bool:
        s = conjecture.lower()
        if re.search(r':\s*(true|⊤)\s*$', s):
            result.issues.append("Trivially True")
            return True
        if re.search(r':\s*(\w+)\s*=\s*\1\s*$', s):
            result.issues.append("Trivially reflexive (a = a)")
            return True
        stmt_part = conjecture.split(":=")[0] \
            if ":=" in conjecture else conjecture
        if "sorry" in stmt_part.lower():
            result.issues.append("Contains sorry in statement")
            return True
        return False

    def _compute_relevance(self, conjecture: str, target: str) -> float:
        from prover.premise.bm25_retriever import tokenize
        conj_tokens = set(tokenize(conjecture))
        target_tokens = set(tokenize(target))
        if not conj_tokens or not target_tokens:
            return 0.0
        overlap = conj_tokens & target_tokens
        return len(overlap) / len(conj_tokens | target_tokens)

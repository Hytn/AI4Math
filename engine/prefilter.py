"""engine/prefilter.py — L0 语法预过滤器

在证明进入 Lean4 REPL 之前, 用纯语法规则快速排除明显无效的输出。
每条规则在 <10μs 内完成, 可过滤掉 ~90% 的无效输出。

规则是可扩展的: 通过 register() 或从插件 YAML 中加载。
每条规则返回结构化的反馈 (不是简单的 True/False),
让 Agent 知道 *为什么* 被拒绝以及 *如何修复*。
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """预过滤结果"""
    passed: bool
    rule_name: str = ""
    reason: str = ""
    fix_hint: str = ""
    severity: str = "error"  # error, warning, info

    @staticmethod
    def ok() -> FilterResult:
        return FilterResult(passed=True)

    @staticmethod
    def reject(rule: str, reason: str, hint: str = "",
               severity: str = "error") -> FilterResult:
        return FilterResult(
            passed=False, rule_name=rule,
            reason=reason, fix_hint=hint, severity=severity)


class FilterRule:
    """过滤规则基类"""
    name: str = "base_rule"
    description: str = ""

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        return FilterResult.ok()


class SorryDetector(FilterRule):
    """检测 sorry/admit — 这些不构成有效证明"""
    name = "sorry_detector"
    description = "Reject proofs containing sorry or admit"

    _PATTERN = re.compile(
        r'\b(sorry|admit)\b'
        r'(?!\s*--)'    # 不匹配注释中的 sorry
    )

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        # 去掉注释行
        lines = [l for l in proof.split("\n")
                 if not l.strip().startswith("--")]
        clean = "\n".join(lines)

        match = self._PATTERN.search(clean)
        if match:
            return FilterResult.reject(
                self.name,
                f"Proof contains `{match.group(0)}` which is not a valid proof term",
                "Remove all `sorry` and `admit`. Every goal must be closed "
                "by a real tactic (simp, ring, omega, exact, apply, etc.)")
        return FilterResult.ok()


class EmptyProofDetector(FilterRule):
    """检测空证明"""
    name = "empty_proof"
    description = "Reject empty or whitespace-only proofs"

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        if not proof or not proof.strip():
            return FilterResult.reject(
                self.name,
                "Proof is empty",
                "Generate a proof starting with `:= by` followed by tactics")
        return FilterResult.ok()


class BracketMatcher(FilterRule):
    """检测括号不匹配"""
    name = "bracket_matcher"
    description = "Reject proofs with mismatched brackets"

    _PAIRS = {"(": ")", "[": "]", "{": "}", "⟨": "⟩"}

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        # 去掉字符串和注释
        clean = re.sub(r'"[^"]*"', '', proof)
        clean = re.sub(r'--[^\n]*', '', clean)

        stack = []
        for i, ch in enumerate(clean):
            if ch in self._PAIRS:
                stack.append((ch, i))
            elif ch in self._PAIRS.values():
                if not stack:
                    return FilterResult.reject(
                        self.name,
                        f"Unmatched closing bracket `{ch}` at position {i}",
                        "Check bracket pairing in the proof")
                open_ch, _ = stack.pop()
                if self._PAIRS[open_ch] != ch:
                    return FilterResult.reject(
                        self.name,
                        f"Mismatched brackets: `{open_ch}` opened but `{ch}` closes",
                        f"Replace `{ch}` with `{self._PAIRS[open_ch]}`")
        if stack:
            open_ch, pos = stack[-1]
            return FilterResult.reject(
                self.name,
                f"Unclosed bracket `{open_ch}` at position {pos}",
                f"Add closing `{self._PAIRS[open_ch]}`")
        return FilterResult.ok()


class Lean3Detector(FilterRule):
    """检测 Lean3 语法 (常见 LLM 错误)"""
    name = "lean3_syntax"
    description = "Detect Lean3 syntax that won't work in Lean4"

    _LEAN3_PATTERNS = [
        # 只匹配独立的小写类型名 (前后不可是标识符字符或点号)
        # 避免匹配 "concatenate" 中的 "nat" 或 "interval" 中的 "int"
        (r'(?<![a-zA-Z0-9_\.])nat(?![a-zA-Z0-9_\.])', "nat", "Nat"),
        (r'(?<![a-zA-Z0-9_\.])int(?![a-zA-Z0-9_\.])', "int", "Int"),
        (r'(?<![a-zA-Z0-9_\.])list(?![a-zA-Z0-9_\.])', "list", "List"),
        (r'(?<![a-zA-Z0-9_\.])bool(?![a-zA-Z0-9_\.])', "bool", "Bool"),
        # Lean3 命名空间风格: nat.xxx, int.xxx, list.xxx
        # 这是 LLM 最常见的 Lean3 错误 — 使用小写前缀的引理名
        (r'(?<![a-zA-Z0-9_])nat\.(?=[a-z])', "nat.xxx", "Nat.xxx"),
        (r'(?<![a-zA-Z0-9_])int\.(?=[a-z])', "int.xxx", "Int.xxx"),
        (r'(?<![a-zA-Z0-9_])list\.(?=[a-z])', "list.xxx", "List.xxx"),
        # begin/end 只在行首 (可选空白后) 匹配, 避免匹配标识符中的子串
        (r'(?:^|\n)\s*begin\s*$', "begin...end", ":= by ... (tactic block)"),
        (r'(?:^|\n)\s*end\s*$', "begin...end", ":= by ... (tactic block)"),
        (r'#check\b', "#check", "remove #check from proof"),
        # lambda 只匹配独立关键字, 不匹配 "lambda_expr" 等标识符
        (r'(?<![a-zA-Z0-9_])lambda(?![a-zA-Z0-9_])', "lambda", "fun"),
        # 移除 ∀ 规则: `∀ x, P x` 在 Lean4 中是合法语法 (类型可推断)
    ]

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        for pattern, old, new in self._LEAN3_PATTERNS:
            if re.search(pattern, proof):
                return FilterResult.reject(
                    self.name,
                    f"Lean3 syntax detected: `{old}`",
                    f"Use Lean4 equivalent: `{new}`",
                    severity="warning")
        return FilterResult.ok()


class NatSubtractGuard(FilterRule):
    """检测 ℕ 减法陷阱 — Lean4 中 ℕ 减法截断到 0"""
    name = "nat_subtract_guard"
    description = "Warn when using ℕ subtraction without ≤ guard"

    _SUB = re.compile(r'(\w+)\s*-\s*(\w+)')
    _SAFE = re.compile(
        r'tsub_add_cancel|Nat\.sub_add_cancel|'
        r'Nat\.sub_le|omega|≤|le_of|Nat\.le',
    )

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        # 只在定理涉及 ℕ 时检查
        if not re.search(r'\bNat\b|: ℕ|: Nat\b', theorem):
            return FilterResult.ok()

        # 减法可能出现在定理声明或证明中 — 两者都需要检查
        combined = f"{theorem}\n{proof}"
        if self._SUB.search(combined) and not self._SAFE.search(proof):
            return FilterResult.reject(
                self.name,
                "Natural number subtraction without ≤ guard. "
                "In Lean4, ℕ subtraction truncates to 0.",
                "Prove `minuend ≥ subtrahend` first, then use "
                "`Nat.sub_add_cancel` or `tsub_add_cancel_of_le`. "
                "Alternative: cast to ℤ with `Int.ofNat`.",
                severity="warning")
        return FilterResult.ok()


class RingOnNatGuard(FilterRule):
    """检测 ring 在 ℕ 减法上的误用"""
    name = "ring_on_nat_sub"
    description = "Warn when using `ring` with ℕ subtraction"

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        if not re.search(r'\bNat\b|: ℕ|: Nat\b', theorem):
            return FilterResult.ok()
        # ring 出现在 proof 中, 减法出现在 proof 或 theorem 中
        combined = f"{theorem}\n{proof}"
        if re.search(r'\bring\b', proof) and re.search(r'\b\w+\s*-\s*\w+', combined):
            return FilterResult.reject(
                self.name,
                "`ring` does not handle ℕ subtraction correctly "
                "(truncated subtraction is not a ring operation)",
                "Use `omega` for linear ℕ arithmetic, or cast to ℤ first",
                severity="warning")
        return FilterResult.ok()


class TacticExistence(FilterRule):
    """检测不存在的 tactic 名称"""
    name = "tactic_existence"
    description = "Detect non-existent tactic names"

    _KNOWN_TACTICS = {
        "simp", "ring", "omega", "norm_num", "linarith", "decide",
        "exact", "apply", "intro", "intros", "rfl", "rw", "rewrite",
        "cases", "induction", "constructor", "contradiction", "exfalso",
        "have", "let", "show", "calc", "conv", "ext", "funext",
        "push_neg", "by_contra", "by_cases", "rcases", "obtain",
        "refine", "use", "exists", "left", "right", "trivial",
        "assumption", "aesop", "tauto", "field_simp", "ring_nf",
        "simp_all", "positivity", "gcongr", "rel", "norm_cast",
        "push_cast", "exact?", "apply?", "rw?", "simp?",
        "sorry", "admit",
    }

    # 关键词: Lean4 语法元素, 不是 tactic 名
    _KEYWORDS = {
        "all", "with", "at", "only", "using", "from", "this",
        "true", "false", "def", "theorem", "lemma", "where",
        "match", "if", "then", "else", "do", "return", "fun",
        "for", "in", "by", "import", "open", "section", "end",
        "namespace", "variable", "example", "instance", "class",
        "structure", "deriving", "private", "protected", "noncomputable",
    }

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        # 提取 tactic 名称 (by 块中每行开头的标识符)
        tactic_pattern = re.compile(
            r'(?:^|\n)\s+(?:·\s*)?([a-z_][a-z_0-9?!]*)',
        )
        for match in tactic_pattern.finditer(proof):
            name = match.group(1)
            if (name not in self._KNOWN_TACTICS
                    and name not in self._KEYWORDS
                    and not name.startswith("simp_")
                    and not name.startswith("norm_")
                    and len(name) > 2):
                return FilterResult.reject(
                    self.name,
                    f"Possibly unknown tactic `{name}`",
                    f"Check spelling. Common tactics: simp, ring, omega, "
                    f"linarith, norm_num, exact, apply, intro, cases, "
                    f"induction, aesop",
                    severity="warning")
        return FilterResult.ok()


class PreFilter:
    """L0 预过滤器: 可扩展的规则引擎

    Usage::

        pf = PreFilter()
        result = pf.check(proof_code, theorem_statement)
        if not result.passed:
            print(f"Rejected by {result.rule_name}: {result.reason}")
            print(f"Fix: {result.fix_hint}")
    """

    def __init__(self, strict: bool = False):
        self.strict = strict  # strict 模式下 warning 也算 reject
        self._rules: list[FilterRule] = []
        self._register_builtins()

    def _register_builtins(self):
        """注册内置规则"""
        self._rules = [
            EmptyProofDetector(),
            SorryDetector(),
            BracketMatcher(),
            Lean3Detector(),
            NatSubtractGuard(),
            RingOnNatGuard(),
            TacticExistence(),
        ]

    def register(self, rule: FilterRule):
        """注册自定义规则"""
        self._rules.append(rule)

    def check(self, proof: str, theorem: str = "") -> FilterResult:
        """运行所有规则, 返回第一个失败结果 (或 OK)

        Warning 级别的规则不阻止验证 (除非 strict=True),
        但会将提示信息附加到结果中供 Agent 参考。
        """
        warnings = []

        for rule in self._rules:
            try:
                result = rule.check(proof, theorem)
                if not result.passed:
                    if result.severity == "error":
                        return result
                    elif result.severity == "warning":
                        if self.strict:
                            return result
                        warnings.append(result)
            except Exception as e:
                logger.warning(f"PreFilter rule {rule.name} raised: {e}")

        # 所有规则通过, 但可能有 warning
        if warnings:
            # 返回 OK 但带上 warning 信息
            combined = FilterResult.ok()
            combined.fix_hint = " | ".join(
                f"[{w.rule_name}] {w.fix_hint}" for w in warnings)
            return combined

        return FilterResult.ok()

    def check_all(self, proof: str, theorem: str = "") -> list[FilterResult]:
        """运行所有规则, 返回全部结果 (用于详细诊断)"""
        results = []
        for rule in self._rules:
            try:
                results.append(rule.check(proof, theorem))
            except Exception as e:
                logger.warning(f"PreFilter rule {rule.name} raised: {e}")
        return results

    def list_rules(self) -> list[dict]:
        return [{"name": r.name, "description": r.description}
                for r in self._rules]

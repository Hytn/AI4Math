"""knowledge/goal_normalizer.py — Goal pattern 规范化

将 Lean4 goal 字符串转化为可比较、可检索的规范化模式。

三级规范化 (Phase 1 实现 Level 1, Level 2/3 后续迭代):

  Level 1 (变量抹除):
    "n m : ℕ, h : n ≤ m ⊢ n + (m - n) = m"
    → "_ _ : ℕ, _ : _ ≤ _ ⊢ _ + (_ - _) = _"

  Level 2 (结构骨架, Phase 4):
    → "ℕ_var, ℕ_var, ℕ_le_hyp ⊢ ℕ_add(ℕ_sub) = ℕ_var"

  Level 3 (语义标签, Phase 5):
    → "nat_arithmetic::subtraction_cancellation"
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════
# Level 1: 变量抹除 (regex-based, Phase 1)
# ═══════════════════════════════════════════════════════════════

# 已知 Lean4 类型名和常用命名空间，不应被抹除
_PRESERVED_TOKENS = frozenset({
    # 基本类型
    "Nat", "Int", "ℕ", "ℤ", "ℚ", "ℝ", "ℂ",
    "Bool", "Prop", "True", "False", "Type",
    "List", "Array", "Option", "String", "Fin", "Unit",
    # 代数结构
    "Group", "Ring", "Field", "Module", "Subgroup",
    "CommRing", "CommGroup", "Monoid", "Semiring",
    # 序/拓扑
    "Set", "Finset", "Multiset", "Filter", "Topology",
    # 关键字
    "fun", "let", "by", "with", "match", "if", "then", "else",
    "forall", "exists", "∀", "∃", "∈", "∉", "⊆", "⊂",
    "∧", "∨", "¬", "→", "↔", "≤", "≥", "<", ">", "≠",
    # 常用运算
    "HAdd", "HSub", "HMul", "HDiv", "HPow", "HMod",
    # sorry / placeholder
    "sorry",
})

# 变量名模式：单个小写字母，或小写字母+数字/下标
_VAR_PATTERN = re.compile(
    r'\b([a-z][a-z0-9_]*\'?)\b'
)

# 数字字面量
_NUM_PATTERN = re.compile(r'\b\d+\b')


def normalize_level1(goal: str) -> str:
    """Level 1 规范化：抹除变量名和数字字面量

    保留类型名、运算符、结构关键字。
    将所有用户定义的变量替换为 '_'。

    Args:
        goal: 原始 Lean4 goal 字符串 (可以带或不带 ⊢ 前缀)

    Returns:
        规范化后的 pattern 字符串
    """
    if not goal or not goal.strip():
        return ""

    result = goal.strip()

    # 抹除数字字面量
    result = _NUM_PATTERN.sub("_N", result)

    # 抹除变量名 (保留已知类型名)
    def _replace_var(m: re.Match) -> str:
        token = m.group(1)
        if token in _PRESERVED_TOKENS:
            return token
        # 保留以大写开头的（类型/构造器名）
        if token[0].isupper():
            return token
        return "_"

    result = _VAR_PATTERN.sub(_replace_var, result)

    # 压缩连续空白
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def normalize_goal_for_key(goal: str) -> str:
    """生成用于 tactic_effectiveness 表的 goal_pattern key

    比 Level 1 更激进地压缩：
    - 截断到前 200 字符 (避免超长 goal 导致的键爆炸)
    - 移除假设名 (保留假设类型)
    """
    pattern = normalize_level1(goal)

    # 进一步压缩假设部分
    # "_ : ℕ, _ : _ ≤ _" → "ℕ, _ ≤ _"
    pattern = re.sub(r'_\s*:\s*', '', pattern)

    # 截断
    if len(pattern) > 200:
        pattern = pattern[:200] + "…"

    return pattern


# ═══════════════════════════════════════════════════════════════
# Domain 分类 (规则, Phase 1)
# ═══════════════════════════════════════════════════════════════

_DOMAIN_SIGNALS: list[tuple[str, list[str]]] = [
    ("number_theory", ["Nat.Prime", "Nat.dvd", "Nat.gcd", "∣", "Prime",
                        "Nat.Coprime", "ZMod", "Int.emod"]),
    ("algebra", ["Ring", "Field", "Group", "Monoid", "CommRing",
                 "Polynomial", "MvPolynomial", "Ideal"]),
    ("analysis", ["Real", "ℝ", "Filter", "Tendsto", "ContinuousOn",
                  "Differentiable", "MeasureTheory", "Metric"]),
    ("combinatorics", ["Finset", "Fintype", "Multiset", "Nat.choose",
                       "card", "Finset.sum"]),
    ("topology", ["TopologicalSpace", "IsOpen", "IsClosed", "Compact",
                  "Connected", "Homeomorph"]),
    ("linear_algebra", ["Module", "LinearMap", "Submodule", "Matrix",
                        "Basis", "Span"]),
    ("order_theory", ["PartialOrder", "Lattice", "Sup", "Inf",
                      "OrderIso", "GaloisConnection"]),
    ("nat_arithmetic", ["ℕ", "Nat", "Nat.add", "Nat.sub", "Nat.mul",
                        "Nat.succ", "Nat.zero"]),
    ("set_theory", ["Set", "Set.mem", "∈", "⊆", "∪", "∩",
                    "Set.Finite", "Set.Infinite"]),
    ("logic", ["Prop", "True", "False", "And", "Or", "Not",
               "Iff", "Exists", "∀", "∃", "¬"]),
]


def classify_domain(goal: str, theorem: str = "") -> str:
    """根据 goal 和 theorem 文本推断数学领域

    返回最匹配的领域标签，或 "general"。
    """
    text = f"{theorem} {goal}".lower()

    scores: dict[str, int] = {}
    for domain, signals in _DOMAIN_SIGNALS:
        score = sum(1 for s in signals if s.lower() in text)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)


# ═══════════════════════════════════════════════════════════════
# 关键词提取 (用于引理检索)
# ═══════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "for", "of", "in", "to",
    "by", "with", "at", "from", "and", "or", "not", "that",
    "this", "it", "all", "any", "some", "have", "has",
    # Lean keywords
    "theorem", "lemma", "def", "example", "instance",
    "where", "let", "fun", "show", "calc", "have", "suffices",
    "import", "open", "namespace", "section", "variable",
})

_KEYWORD_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_.]+')


def extract_keywords(text: str) -> list[str]:
    """从 theorem/lemma 文本中提取检索关键词"""
    tokens = _KEYWORD_PATTERN.findall(text)
    keywords = []
    seen = set()
    for t in tokens:
        lower = t.lower()
        if lower not in _STOP_WORDS and lower not in seen and len(t) > 1:
            seen.add(lower)
            keywords.append(t)
    return keywords[:30]  # 限制数量


def statement_hash(statement: str) -> str:
    """引理 statement 的去重哈希"""
    normalized = re.sub(r'\s+', ' ', statement.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

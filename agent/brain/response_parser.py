"""agent/brain/response_parser.py — 从 LLM 输出中提取结构化内容"""
from __future__ import annotations
import re
import json


_LEAN_KEYWORDS = (
    "by\n",
    ":= by",
    "\nby",
    "theorem ",
    "lemma ",
    "example ",
    "\n  exact ",
    "\n  simp",
    "\n  rw ",
    "\n  intro",
    "\n  apply ",
    "\n  have ",
)

_NARRATIVE_MARKERS = (
    "we need to",
    "let's",
    "actually",
    "therefore",
    "thus",
    "in conclusion",
    "the idea is",
)

# 证明块后常见「收尾句」，应从 unfenced 推理链尾部剥掉，避免 Lean 当战术解析。
_TRAILING_NARRATIVE_LINE = re.compile(
    r"(?i)^\s*("
    r"in conclusion|therefore|thus|hence|note that|we have shown|we've shown|"
    r"this completes|qed\.?|so we are done|this proves|the proof is complete|"
    r"overall|finally|to summarize|in summary"
    r")\b"
)

# 行首看起来像 Lean（证明草稿 / tactic），用于从混合中英尾部裁剪、污染检测。
_LEAN_LINE_HEAD = re.compile(
    r"^\s*(?:"
    r"have\b|let\b|exact\b|exact_mod_cast\b|simpa?\b|simp\b|rw\b|apply\b|intro\b|"
    r"intros?\b|calc\b|show\b|theorem\b|lemma\b|example\b|by\b|·|case\b|"
    r"omega\b|ring\b|linarith\b|norm_num\b|field_simp\b|native_decide\b|subst\b|"
    r"aesop\b|constructor\b|refine\b|use\b|split\b|left\b|right\b|push_cast\b|"
    r"convert\b|congr\b|congr_arg\b|congr_fun\b|ext\b|funext\b|sorry\b|done\b|"
    r"by_cases\b|by_contra\b|contradiction\b|cases\b|rcases\b|obtain\b|"
    r"rintro\b|revert\b|generalize\b|specialize\b|haveI\b|letI\b|"
    r"decide\b|trivial\b|exact\?\b|rfl\b|"
    r"import\b|open\b|variable\b|variables\b|set_option\b|section\b|end\b|"
    r"#align\b|#eval\b|#check\b|instance\b|class\b|def\b|structure\b|inductive\b|"
    r"match\b|if\b|then\b|else\b|try\b|fail\b|repeat\b|first\b|all_goals\b|"
    r"#\||--|/\*"
    r")"
)

_STRUCTURAL_LEAN_HINT = re.compile(
    r"(?m)(?:"
    r":=\s*by\b|:=\s*$|^\s*by\b|^\s*have\b|^\s*theorem\b|^\s*lemma\b|^\s*example\b|"
    r"^\s*exact\b|^\s*simpa?\b|^\s*exact_mod_cast\b|^\s*calc\b|^\s*rw\b"
    r")"
)

# 缩进块内常见 tactic 开头（避免把 ``  Let's try`` 当成 Lean 延续行）
_INDENTED_TACTIC_HEAD = re.compile(
    r"(?xi)^(?:have\b|let\b|exact\b|exact_mod_cast\b|simpa?\b|simp\b|rw\b|apply\b|"
    r"intro\b|intros?\b|calc\b|show\b|by\b|·|cases\b|rcases\b|rintro\b|"
    r"linarith\b|ring\b|omega\b|norm_num\b|field_simp\b|native_decide\b|subst\b|"
    r"constructor\b|refine\b|use\b|split\b|left\b|right\b|push_cast\b|convert\b|"
    r"congr\b|ext\b|funext\b|sorry\b|done\b|aesop\b|obtain\b|revert\b|"
    r"generalize\b|specialize\b|by_cases\b|by_contra\b|contradiction\b|"
    r"iterate\b|try\b|fail\b|first\b|all_goals\b)\b"
)

_GATE_PROSE_MARKERS = (
    "we need to prove",
    "in conclusion",
    "the idea is",
    "this is more explicit",
)


def looks_like_lean_line(line: str) -> bool:
    """Heuristic: non-empty line is tactic / declaration / comment / continuation."""
    if not line.strip():
        return True
    t = line.strip()
    if t.startswith("--"):
        return True
    if t.startswith("/--") or ":= by" in line:
        return True
    if _LEAN_LINE_HEAD.match(line):
        return True
    if re.match(r"^\s{2,}\S", line):
        st = line.strip()
        if _INDENTED_TACTIC_HEAD.match(st):
            return True
        if st.startswith("--") or st.startswith("·"):
            return True
        if ":=" in line or "⟨" in line or "∀" in line or "∃" in line:
            return True
        if "`" in line and (":=" in line or "by " in line):
            return True
        return False
    if re.match(r"^\s*·\s*\S", line):
        return True
    # calc 步骤
    if re.match(r"^\s*_\s+", line):
        return True
    # 仅括号
    if re.fullmatch(r"[\s\)\]\}⟨⟩:,]+", line):
        return True
    return False


def _trim_trailing_nonlean_lines(lean: str) -> str:
    """去掉末尾连续「不像 Lean」的行（模型在战术块后继续写英文解说）。"""
    lines = (lean or "").rstrip().split("\n")
    while lines:
        raw = lines[-1]
        if not raw.strip():
            lines.pop()
            continue
        if looks_like_lean_line(raw):
            break
        lines.pop()
    return "\n".join(lines)


def _strip_inline_fence_artifacts(lean: str) -> str:
    """去掉残留的 markdown fence 行或孤立 ```。"""
    out_lines = []
    for line in (lean or "").split("\n"):
        if re.match(r"^\s*```(?:lean|lean4)?\s*$", line, re.I):
            continue
        if line.strip() == "```":
            continue
        line = line.replace("```", "")
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _trim_trailing_narrative_lines(lean: str) -> str:
    """Drop English wrap-up lines after an unfenced proof lifted from long CoT."""
    lines = (lean or "").rstrip().split("\n")
    while lines:
        raw = lines[-1]
        last = raw.strip()
        if not last:
            lines.pop()
            continue
        if _TRAILING_NARRATIVE_LINE.match(last):
            lines.pop()
            continue
        break
    return "\n".join(lines)


def _sanitize_candidate(cand: str) -> str:
    c = (cand or "").strip()
    if not c:
        return ""
    c = re.sub(r"(?m)^\s*```(?:lean|lean4)?\s*$", "", c)
    c = re.sub(r"(?m)^\s*```\s*$", "", c)
    return c.strip()


def _finalize_extracted_body(cand: str) -> str:
    c = _sanitize_candidate(cand)
    if not c:
        return ""
    c = _strip_inline_fence_artifacts(c)
    c = _trim_trailing_narrative_lines(c)
    c = _trim_trailing_nonlean_lines(c)
    return c.rstrip()


def _score_lean_block(text: str) -> int:
    score = 0
    for line in text.split("\n"):
        if not line.strip():
            continue
        if looks_like_lean_line(line):
            score += 3
        else:
            score -= 2
    if _STRUCTURAL_LEAN_HINT.search(text):
        score += 5
    return score


def _collect_fence_candidates(s: str) -> list[str]:
    cands: list[str] = []
    for pattern in [
        r"```lean\s*([\s\S]*?)```",
        r"```lean4\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]:
        for m in re.finditer(pattern, s, re.IGNORECASE):
            c = _sanitize_candidate(m.group(1))
            if c:
                cands.append(c)
    return cands


def clean_extracted_lean_for_gate(lean: str) -> str:
    """供校验门禁使用：去 fence 残留、尾部叙事、尾部英文解说。"""
    return _finalize_extracted_body(lean or "")


def prose_pollution_lines_remain(lean: str) -> bool:
    """在非 Lean 行上检测明显散文污染（避免全文子串误杀合法 tactic 块）。"""
    s = lean or ""
    for line in s.split("\n"):
        if looks_like_lean_line(line):
            continue
        low = line.lower()
        if any(m in low for m in _GATE_PROSE_MARKERS):
            return True
    return False


def is_probably_lean_proof(cand: str) -> bool:
    c = (cand or "").strip()
    if not c:
        return False
    if "```" in c:
        return False
    has_kw = any(k in c.lower() for k in _LEAN_KEYWORDS)
    has_struct = bool(_STRUCTURAL_LEAN_HINT.search(c))
    if not has_kw and not has_struct:
        return False
    nonlean_narrative_hits = 0
    for line in c.split("\n"):
        if looks_like_lean_line(line):
            continue
        low = line.lower()
        if any(m in low for m in _NARRATIVE_MARKERS):
            nonlean_narrative_hits += 1
    if nonlean_narrative_hits >= 3 and c.count("\n") < 20:
        return False
    return True


def is_extract_polluted_for_verify_gate(lean: str) -> bool:
    """与 ProofLoop 门禁一致：空、不像 Lean、或残留明显散文行。"""
    s = clean_extracted_lean_for_gate(lean)
    if not s.strip():
        return True
    if "```" in s:
        return True
    if not is_probably_lean_proof(s):
        return True
    if prose_pollution_lines_remain(s):
        return True
    return False


def extract_lean_code(response: str, allow_unfenced: bool = True) -> str:
    """Pull Lean proof text from an LLM message.

    Handles ```lean / ```lean4 / generic ``` fences (with or without newline after tag),
    then markdown-stripped text, then unfenced `:= by` / `theorem` heuristics.
    """
    if not (response and response.strip()):
        return ""
    s = response.strip()
    fence_cands = _collect_fence_candidates(s)
    if fence_cands:
        best = max(fence_cands, key=lambda t: (_score_lean_block(t), len(t)))
        if _score_lean_block(best) < 1 and fence_cands:
            best = fence_cands[-1]
        fin = _finalize_extracted_body(best)
        if fin:
            return fin
    if not allow_unfenced:
        return ""
    matches = list(re.finditer(r":=\s*by\b", s))
    for m in reversed(matches):
        cand = _finalize_extracted_body(s[m.start() :])
        if cand and is_probably_lean_proof(cand):
            return cand
    if matches:
        cand = _finalize_extracted_body(s[matches[-1].start() :])
        if cand:
            return cand if is_probably_lean_proof(cand) else ""
    by_matches = list(re.finditer(r"(?m)^\s*by\b", s))
    for m in reversed(by_matches):
        cand = _finalize_extracted_body(s[m.start() :])
        if cand and is_probably_lean_proof(cand):
            return cand
    decl_pat = r"(?m)^\s*(?:theorem|lemma|example)\s+\S"
    decl_matches = list(re.finditer(decl_pat, s))
    for m in reversed(decl_matches):
        tail = _finalize_extracted_body(s[m.start() :])
        if len(tail) >= 8000 or tail.count("```") >= 2:
            continue
        if re.search(
            r"(by\s|exact |simp |ring|linarith|field_simp|have |apply )", tail
        ):
            if is_probably_lean_proof(tail):
                return tail
    loose_fence_matches = list(re.finditer(r"```\s*\n(.*?)\n```", s, re.DOTALL | re.IGNORECASE))
    for m in reversed(loose_fence_matches):
        cand = _finalize_extracted_body(m.group(1))
        if cand and is_probably_lean_proof(cand):
            return cand
    lines = s.split('\n')
    lean_blocks = []
    current_block = []
    in_block = False
    for line in lines:
        if looks_like_lean_line(line):
            if not in_block:
                in_block = True
                current_block = []
            current_block.append(line)
        else:
            if in_block:
                if len(current_block) > 2: 
                    lean_blocks.append('\n'.join(current_block))
                in_block = False
    if in_block and len(current_block) > 2:
        lean_blocks.append('\n'.join(current_block))
    for block in reversed(lean_blocks):
        if is_probably_lean_proof(block):
            return block
    return ""


def extract_lean_thinking_fallback(thinking: str) -> str:
    """从推理链中取 Lean：先围栏块，再允许 unfenced ``:= by`` / ``theorem``；去掉尾部叙述句。

    DeepSeek 等模型常把完整证明写在 ``reasoning_content`` 且无 ```lean```，
    旧逻辑 ``allow_unfenced=False`` 会导致整段为空。
    """
    t = thinking or ""
    if not t.strip():
        return ""
    c = extract_lean_code(t, allow_unfenced=False)
    if c.strip():
        return _finalize_extracted_body(c)
    c = extract_lean_code(t, allow_unfenced=True)
    if not c.strip():
        return ""
    c = _finalize_extracted_body(c)
    return c if is_probably_lean_proof(c) else ""


def extract_lean_from_model_output(content: str, thinking: str = "") -> str:
    """Prefer ``content``; if empty, use fenced then unfenced proof from ``thinking``."""
    a = extract_lean_code(content or "", allow_unfenced=True)
    if a.strip():
        return a
    return extract_lean_thinking_fallback(thinking or "")


def extract_json(response: str) -> dict:
    try:
        m = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
        return json.loads(m.group(1) if m else response)
    except (json.JSONDecodeError, AttributeError):
        return {}


def extract_sorry_blocks(lean_code: str) -> list[dict]:
    blocks = []
    for i, line in enumerate(lean_code.split("\n")):
        if "sorry" in line:
            blocks.append({"line": i + 1, "content": line.strip()})
    return blocks

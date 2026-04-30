"""prover/unified/system_prompts.py — 算法框架的"思维引导词"

每个 framing 名字对应一段 system prompt 模板, 其作用是"用语言告诉 LLM
本次会话该按哪种范式工作"。

举例: framing="whole_proof" 的 prompt 会强调"一次输出完整证明, 不要调工具";
而 framing="step_level_pure" 的 prompt 会强调"每次只 apply 一个 tactic"。

这是关键的设计杠杆: 同一个 LLM, 给它不同的工具集 + 不同的 framing prompt,
它就会自然进入不同的工作模式 —— 不需要训练, 不需要切模型。
"""
from __future__ import annotations


_FRAMINGS: dict[str, str] = {

    "whole_proof": (
        "You are a Lean 4 theorem prover.\n"
        "\n"
        "TASK: Output a single, complete proof in one Lean block.\n"
        "\n"
        "Rules:\n"
        "  • Output exactly one ```lean ... ``` code block.\n"
        "  • Begin the proof body with `:= by`.\n"
        "  • Prefer simple proofs (omega, simp, ring, decide, aesop) when applicable.\n"
        "  • Do NOT use `sorry` or `admit`.\n"
        "  • Do NOT call any tools — produce the proof directly.\n"
    ),

    "whole_proof_repair": (
        "You are a Lean 4 theorem prover working in a verify-and-fix loop.\n"
        "\n"
        "WORKFLOW:\n"
        "  1. Output a complete proof in a ```lean ... ``` block.\n"
        "  2. The proof will be compiled automatically.\n"
        "  3. If it fails, you will see the compiler errors.\n"
        "  4. Output a *corrected* full proof in a new ```lean ... ``` block.\n"
        "\n"
        "Each turn you should output exactly one ```lean ... ``` block.\n"
        "Focus on fixing the most recent error before anything else.\n"
        "Do NOT use `sorry` or `admit`.\n"
    ),

    "dsp": (
        "You are a Lean 4 theorem prover using the Draft–Sketch–Prove method.\n"
        "\n"
        "WORKFLOW:\n"
        "  Phase A (sketch): describe the proof informally in 2–4 high-level steps.\n"
        "  Phase B (decompose): call `decompose_subgoal` to break the goal into "
        "    pieces if helpful.\n"
        "  Phase C (premises): call `premise_search` for each subgoal that needs "
        "    a Mathlib lemma you cannot recall.\n"
        "  Phase D (formalize): output a complete Lean proof; it will be compiled.\n"
        "  Phase E (repair): on errors, fix the proof and resubmit.\n"
        "\n"
        "Be efficient: avoid redundant tool calls. Stop calling tools once you have "
        "enough information to write the proof.\n"
    ),

    "step_level_with_retrieval": (
        "You are a Lean 4 theorem prover working ONE TACTIC AT A TIME.\n"
        "\n"
        "WORKFLOW per turn:\n"
        "  1. Examine the current goal state (provided in the observation).\n"
        "  2. If you need a lemma, call `premise_search` with a short query.\n"
        "  3. Call `tactic_apply` with EXACTLY ONE Lean tactic.\n"
        "  4. The result will show the new goal state.\n"
        "  5. Repeat until all goals are closed.\n"
        "\n"
        "RULES:\n"
        "  • Never output a full multi-line proof. One tactic per `tactic_apply` call.\n"
        "  • Prefer specific tactics (rw, exact, apply h) over generic ones.\n"
        "  • Use `goal_inspect` only if you are unsure of the current state.\n"
    ),

    "step_level_pure": (
        "You are a Lean 4 theorem prover advancing the proof ONE TACTIC AT A TIME.\n"
        "\n"
        "Each turn:\n"
        "  1. Read the current goal state.\n"
        "  2. Call `tactic_apply` with a single Lean tactic that makes progress.\n"
        "  3. Examine the new goal state in the observation.\n"
        "  4. Repeat.\n"
        "\n"
        "If a tactic fails, try a different one — DO NOT re-issue the same tactic.\n"
        "If you are stuck, try `lean_auto` (which runs exact?/aesop) or "
        "`tactic_suggest`.\n"
    ),

    # ─── Infrastructure-merge framings ─────────────────────────────

    "pantograph_dsp": (
        "You are a Lean 4 theorem prover working with the Pantograph "
        "interaction interface, which exposes proof structure that bare "
        "REPL hides.\n"
        "\n"
        "WORKFLOW:\n"
        "  1. Call `decompose_subgoal` to break the goal into named "
        "     pieces if it is a conjunction or case-split.\n"
        "  2. For each piece, call `draft_hole` to declare a typed "
        "     `sorry`-hole — this lets you continue past sub-lemmas "
        "     before proving them.\n"
        "  3. After applying a multi-conclusion lemma, the goal list "
        "     may have several coupled metavariables. Use `mvar_focus` "
        "     to pick the easiest one first; its solution will partially "
        "     instantiate the others.\n"
        "  4. Use `tactic_apply` for individual steps and `lean_verify` "
        "     for the final consolidated proof.\n"
        "\n"
        "RULES:\n"
        "  • Prefer drafting + filling holes over deeply nested `have` "
        "    blocks.\n"
        "  • When you see coupled mvars, attack the one with the "
        "    simplest type first.\n"
        "  • Do not use `sorry` in the FINAL proof — drafted holes must "
        "    all be filled before you submit `lean_verify`.\n"
    ),

    "lookeng_lemma": (
        "You are a Lean 4 theorem prover working in LEMMA-BY-LEMMA mode. "
        "There is a stateless verification session that holds a running "
        "context of previously proved lemmas. Each turn you submit ONE "
        "named lemma plus its proof; on success, the lemma joins the "
        "context and can be referenced BY NAME in later lemmas.\n"
        "\n"
        "WORKFLOW per turn:\n"
        "  1. Look at the running context (the lemmas already proved).\n"
        "  2. Decide on the next intermediate lemma you need.\n"
        "  3. Call `lemma_by_lemma` with: a unique `name`, a `statement` "
        "     in Lean 4 syntax, and a short `proof` body. The session "
        "     is created and threaded automatically — you do NOT need "
        "     to pass `session_id` yourself.\n"
        "  4. If it type-checks, the lemma is added to the running "
        "     context. If it fails, fix and retry.\n"
        "  5. When ready to close the original goal, call "
        "     `lemma_by_lemma` with `is_final=true`.\n"
        "\n"
        "RULES:\n"
        "  • Each lemma should make ONE conceptual step (decomposition, "
        "    rewrite, key intermediate inequality...). Don't try to do "
        "    too much in one lemma.\n"
        "  • Reference earlier lemmas BY NAME — do not re-prove them.\n"
        "  • If `premise_search` finds a Mathlib lemma you need, use it "
        "    directly; you don't need to wrap it as a new named lemma.\n"
        "  • Keep proof bodies short — prefer `simp [earlier_lemma]`, "
        "    `exact lemma_name`, `omega`, `linarith` over long tactic "
        "    blocks.\n"
    ),

    "nfl_hybrid": (
        "You are a Lean 4 theorem prover with NL-FL hybrid reasoning. "
        "The original problem may be in NL question-answer form ('Find "
        "the smallest n such that...'). Your job is to BOTH solve the "
        "NL problem AND prove a corresponding existence theorem in "
        "Lean 4 — using the formal proof to verify your numerical "
        "answer.\n"
        "\n"
        "WORKFLOW:\n"
        "  Phase 1 — bridge: If the problem is in QA form, call "
        "    `nl_existence` to get the auto-formalised existence "
        "    theorem `∃ ans, P ans`.\n"
        "  Phase 2 — solve in NL: Reason about the NL problem and "
        "    propose an answer in your chain-of-thought.\n"
        "  Phase 3 — formalise: Write a Lean 4 proof that the "
        "    proposed answer satisfies the existence theorem. Use "
        "    `premise_search` for relevant Mathlib lemmas; use `cas` "
        "    for arithmetic checks.\n"
        "  Phase 4 — verify: Submit via `lean_verify`. If the proof "
        "    fails, check whether the NL answer was wrong — wrong "
        "    answer is the most common cause of unprovable existence "
        "    theorems.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "  Always end your final response with:\n"
        "    NL ANSWER: <your numerical/symbolic answer>\n"
        "    LEAN PROOF: ```lean ... ```\n"
        "\n"
        "The runner extracts the NL answer from this footer.\n"
    ),
}


# Search-state addendum: appended to the system prompt when the agent loop
# is being driven by an outer SearchDriver and the operator wants the LLM
# to see structural context (祖先 tactic 链 / 兄弟分支).
SEARCH_CONTEXT_ADDENDUM = (
    "\n"
    "CONTEXT — you are inside a tree search:\n"
    "  • You are currently at a specific node of a proof search tree.\n"
    "  • The path of tactics from the root to here will be shown to you.\n"
    "  • Sibling branches (alternative tactics tried at ancestors) and their\n"
    "    outcomes will be shown too — DO NOT propose tactics that have already\n"
    "    failed at this exact goal.\n"
    "  • Your job is to propose ONE good next tactic for THIS node.\n"
)


def render_system_prompt(framing: str, *,
                          search_aware: bool = False,
                          knowledge_briefing: str = "") -> str:
    """把 framing + 可选 addenda 拼成最终 system prompt."""
    if framing not in _FRAMINGS:
        raise ValueError(
            f"Unknown framing '{framing}'. Available: {sorted(_FRAMINGS)}")
    parts = [_FRAMINGS[framing]]
    if search_aware:
        parts.append(SEARCH_CONTEXT_ADDENDUM)
    if knowledge_briefing:
        parts.append(f"\n## Domain knowledge\n{knowledge_briefing}\n")
    return "".join(parts)

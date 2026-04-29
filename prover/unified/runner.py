"""prover/unified/runner.py — 统一证明管线入口

将 Profile 编译成可执行的运行时:

  Profile (声明式开关) ─┐
                        ├─→ AgentLoop (核心)        ─→ dialog.json
  ToolRegistry (按 kit) ─┤   + 可选外部 SearchDriver
  System Prompt ────────┘   + 可选 ObservationInjector

主要 API::

    runner = UnifiedProofRunner(
        llm=async_llm,
        lean_pool=lean_pool,
        knowledge_store=ks,
        retriever=retr,
    )
    result = await runner.run(problem, profile_name="mcts")
    result.save_unified("results/traces/<id>")     # 标准 dialog.json

任何新算法 = 在 profiles.PRESETS 里加一项, 不动 runner 代码。
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from agent.runtime.agent_loop import AgentLoop, LoopConfig, LoopResult
from agent.tools.base import ToolContext
from agent.tools.registry import ToolRegistry
from common.response_parser import extract_lean_code
from prover.verifier.sorry_detector import detect_sorry

from prover.unified.profiles import (
    Profile, get_profile, ToolKit, SearchConfig,
)
from prover.unified.system_prompts import render_system_prompt
from prover.unified.tool_kits import build_tool_registry
from prover.unified.search_driver import (
    SharedSearchState, make_driver,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedResult:
    """A single run's outcome — wraps LoopResult + search summary."""
    profile_name: str
    success: bool
    proof_code: str = ""
    loop_result: Optional[LoopResult] = None
    sub_results: list = field(default_factory=list)  # for parallel mode
    search_summary: dict = field(default_factory=dict)
    total_duration_ms: int = 0

    def save_unified(self, task_dir: str, *, problem_id: str = "",
                     model: str = "", provider: str = "",
                     system_prompt: str = "",
                     tools: list = None,
                     initial_task: str = ""):
        """Save dialog.json — standard project format."""
        if self.loop_result is None:
            logger.warning("No loop_result to save")
            return
        return self.loop_result.save_unified(
            task_dir, problem_id=problem_id, model=model,
            provider=provider, system_prompt=system_prompt,
            tools=tools, initial_task=initial_task,
        )


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

class UnifiedProofRunner:
    """One Profile in, one dialog.json out."""

    def __init__(self, *, llm, lean_pool=None,
                  knowledge_store=None,
                  retriever=None,
                  broadcast_bus=None):
        self.llm = llm
        self.lean_pool = lean_pool
        self.knowledge_store = knowledge_store
        self.retriever = retriever
        self.broadcast_bus = broadcast_bus

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def run(self, problem, *,
                   profile_name: str = "whole_proof_repair",
                   profile: Optional[Profile] = None) -> UnifiedResult:
        """Run a single proof attempt under the given profile."""
        prof = profile or get_profile(profile_name)
        start = time.time()
        logger.info(f"[unified] starting profile='{prof.name}', "
                     f"max_turns={prof.max_turns}, "
                     f"search={prof.search.kind}, "
                     f"tools={[t.value for t in prof.tools]}")

        # ── Dispatch by search kind ──
        if prof.search.kind == "none":
            ur = await self._run_single_loop(problem, prof)
        elif prof.search.kind == "parallel":
            ur = await self._run_parallel(problem, prof)
        elif prof.search.kind in ("best_first", "ucb", "beam"):
            ur = await self._run_with_search(problem, prof)
        else:
            raise ValueError(f"unknown search.kind: {prof.search.kind}")

        ur.total_duration_ms = int((time.time() - start) * 1000)
        return ur

    # ──────────────────────────────────────────────────────────────────
    # Mode A: single AgentLoop, no outer search
    # ──────────────────────────────────────────────────────────────────

    async def _run_single_loop(self, problem, profile: Profile) -> UnifiedResult:
        """Whole-proof / repair / DSP / ReProver / LeanDojo —— 都走这条."""
        registry = build_tool_registry(
            profile,
            lean_pool=self.lean_pool,
            knowledge_store=self.knowledge_store,
            retriever=self.retriever,
            broadcast_bus=self.broadcast_bus,
            search_state=None,
        )

        # Optional knowledge briefing (ReProver 风格还会通过 tool 查; 这里
        # 注入一份静态简报作为开场上下文)
        briefing = ""
        if profile.observation.include_knowledge_briefing \
                and self.knowledge_store:
            briefing = await self._build_briefing(problem)

        system_prompt = render_system_prompt(
            profile.framing,
            search_aware=False,
            knowledge_briefing=briefing)

        tool_ctx = ToolContext(
            agent_name=f"unified.{profile.name}",
            theorem_statement=problem.theorem_statement,
        )

        config = LoopConfig(
            max_turns=profile.max_turns,
            temperature=profile.temperature,
            timeout_seconds=profile.stop.timeout_seconds,
            max_total_tokens=profile.stop.max_total_tokens,
            stop_on_proof=profile.stop.on_proof_found,
            stop_on_text_only=profile.stop.on_text_only,
        )

        loop = self._make_loop(registry, config, profile)

        # v3: 富初始 prompt — 题目 + 检索引理 + few-shot
        initial = self._build_initial_message(problem, profile)

        loop_result = await loop.run(
            system_prompt=system_prompt,
            initial_message=initial,
            tool_ctx=tool_ctx,
        )

        # v3: auto_inject_lean_compile 后置兜底
        # 如果 loop 因 text_only 终止且产出 lean 代码但未走 lean_verify,
        # 这里自动跑一次完整编译, 让 success 标志反映真实验证结果。
        if (profile.observation.auto_inject_lean_compile
                and loop_result.has_proof
                and self.lean_pool is not None
                and "lean_verify" not in (loop_result.tools_called or [])):
            verified = await self._auto_verify_proof(
                problem, loop_result.proof_code)
            if verified is not None:
                loop_result.stopped_reason = (
                    "proof_found" if verified else "verification_failed")

        return UnifiedResult(
            profile_name=profile.name,
            success=loop_result.has_proof
                    and loop_result.stopped_reason == "proof_found",
            proof_code=loop_result.proof_code,
            loop_result=loop_result,
        )

    # ──────────────────────────────────────────────────────────────────
    # Mode B: outer search driver + per-node AgentLoop expansion
    # ──────────────────────────────────────────────────────────────────

    async def _run_with_search(self, problem, profile: Profile) -> UnifiedResult:
        """MCTS / best_first / beam —— driver 调度多次 expansion."""
        # Initialise the shared tree state from the theorem's root goal.
        root_env_id, root_goals = await self._init_root_state(problem)
        state = SharedSearchState(root_env_id=root_env_id,
                                    root_goals=root_goals)

        # Build per-node AgentLoop (registered with state-aware tools)
        registry = build_tool_registry(
            profile,
            lean_pool=self.lean_pool,
            knowledge_store=self.knowledge_store,
            retriever=self.retriever,
            broadcast_bus=self.broadcast_bus,
            search_state=state,        # ← 关键: tools 持有同一 state
        )

        sc: SearchConfig = profile.search
        driver = make_driver(
            sc.kind, state,
            max_nodes=sc.max_nodes,
            max_depth=sc.max_depth,
            expansion_max_turns=sc.expansion_max_turns,
            beam_width=sc.beam_width,
            ucb_c=sc.ucb_c,
        )

        # Each expansion is one AgentLoop call anchored at the chosen node.
        # The loop's tactic_apply tool mutates `state` via the shared object.
        all_loop_results: list[LoopResult] = []

        async def expand_one_node(*, node_id: int, max_turns: int):
            briefing = ""
            if profile.observation.include_knowledge_briefing \
                    and self.knowledge_store:
                briefing = await self._build_briefing(problem)

            system_prompt = render_system_prompt(
                profile.framing,
                search_aware=profile.observation.include_search_state_in_prompt,
                knowledge_briefing=briefing)

            initial = self._build_node_prompt(problem, state, node_id)

            tool_ctx = ToolContext(
                agent_name=f"unified.{profile.name}.node{node_id}",
                theorem_statement=problem.theorem_statement,
            )

            config = LoopConfig(
                max_turns=max_turns,
                temperature=profile.temperature,
                timeout_seconds=30.0,        # per-node budget
                max_total_tokens=20_000,
                stop_on_proof=False,         # search driver decides termination
                stop_on_text_only=True,
            )

            loop = self._make_loop(registry, config, profile)
            lr = await loop.run(
                system_prompt=system_prompt,
                initial_message=initial,
                tool_ctx=tool_ctx,
            )
            all_loop_results.append(lr)

        await driver.run(expand_one_node=expand_one_node)

        # Reconstruct the proof from the solved path
        proof_code = ""
        success = False
        if state.solved_node_id is not None:
            tactics = [
                n.tactic for n in state.ancestors(state.solved_node_id)
                if n.tactic
            ]
            proof_code = self._tactics_to_proof(
                problem.theorem_statement, tactics)
            success = True

        # Merge dialogs from per-node loops into a single LoopResult-shaped view
        merged = self._merge_loops(all_loop_results, proof_code, success)

        return UnifiedResult(
            profile_name=profile.name,
            success=success,
            proof_code=proof_code,
            loop_result=merged,
            search_summary={
                "kind": profile.search.kind,
                "total_nodes": len(state.nodes),
                "max_depth": max(n.depth for n in state.nodes.values()),
                "solved_node": state.solved_node_id,
                "expansions": len(all_loop_results),
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Mode C: parallel — N profiles run side-by-side, broadcast bus shared
    # ──────────────────────────────────────────────────────────────────

    async def _run_parallel(self, problem, profile: Profile) -> UnifiedResult:
        """异构 N 个 sub-profile 并行 + 共享广播总线 (项目原有特色)."""
        sub_names = profile.search.parallel_profiles or [profile.name]
        sub_profiles = [get_profile(n) for n in sub_names]
        # Share broadcast bus across all sub-runs
        if self.broadcast_bus is None:
            try:
                from engine.broadcast import BroadcastBus
                self.broadcast_bus = BroadcastBus()
            except Exception:
                self.broadcast_bus = None

        tasks = [
            self.run(problem, profile=sp.__class__(**{
                **sp.__dict__,
                # Reset search to none for sub-profiles to avoid recursion
                "search": SearchConfig(kind="none"),
            }))
            for sp in sub_profiles
        ]
        sub_results: list[UnifiedResult] = await asyncio.gather(
            *tasks, return_exceptions=False)

        # Pick the first successful one, else best by has_proof
        winner = next((r for r in sub_results if r.success), None)
        if winner is None:
            winner = max(sub_results,
                         key=lambda r: (bool(r.proof_code), r.profile_name))

        return UnifiedResult(
            profile_name=profile.name,
            success=winner.success,
            proof_code=winner.proof_code,
            loop_result=winner.loop_result,
            sub_results=sub_results,
            search_summary={"kind": "parallel",
                            "sub_profiles": sub_names},
        )

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _make_loop(self, registry: ToolRegistry,
                    config: LoopConfig,
                    profile: Profile) -> AgentLoop:
        """Build the loop with on_turn hook for auto-inject behaviors."""
        # Auto-inject is currently implemented via tools (lean_verify exists,
        # tactic_apply auto-returns goal state). For the optional auto-call
        # of lean_verify when LLM produced lean code without invoking it,
        # we'd plug into AgentLoop.on_turn. Kept minimal here.
        return AgentLoop(llm=self.llm, tools=registry, config=config)

    async def _init_root_state(self, problem):
        """Establish a Lean REPL env at the theorem header — root of search."""
        if not self.lean_pool:
            return 0, [problem.theorem_statement]
        try:
            base = getattr(self.lean_pool, "base_env_id", 0)
            # The pool ought to expose a way to set up the theorem context.
            # Falling back gracefully if it doesn't.
            return base, [problem.theorem_statement]
        except Exception as e:
            logger.warning(f"init_root_state fallback: {e}")
            return 0, [problem.theorem_statement]

    def _build_node_prompt(self, problem, state: SharedSearchState,
                            node_id: int) -> str:
        """Per-node user message: shows ancestors and current goal."""
        node = state.nodes[node_id]
        ancestors = state.ancestors(node_id)
        path = " ; ".join(
            a.tactic for a in ancestors if a.tactic) or "(root)"
        goals_text = "\n".join(f"  ⊢ {g}" for g in node.goals) \
            or "  (no goals)"
        failed_hint = ""
        if node.failed_tactics:
            failed_hint = (
                f"\n\nAvoid these tactics — they already failed at this "
                f"goal: {sorted(node.failed_tactics)}")
        return (
            f"Theorem:\n```lean\n{problem.theorem_statement}\n```\n\n"
            f"Tactic path so far (from root): {path}\n"
            f"Current goals at depth {node.depth}:\n{goals_text}"
            f"{failed_hint}\n\n"
            f"Propose ONE tactic for the current goal. Call `tactic_apply` "
            f"with it."
        )

    def _tactics_to_proof(self, theorem: str, tactics: list[str]) -> str:
        """Concat tactic path → Lean proof body."""
        body = "\n  ".join(tactics) if tactics else "sorry"
        # If theorem already ends with ":= by", we splice in the body.
        if ":= by" in theorem:
            return theorem.split(":= by")[0] + ":= by\n  " + body
        return f"{theorem} := by\n  {body}"

    def _merge_loops(self, loops: list[LoopResult],
                       proof_code: str, success: bool) -> LoopResult:
        """Squash N per-node LoopResults into one for dialog.json output."""
        if not loops:
            return LoopResult(content="", proof_code=proof_code,
                              stopped_reason=("proof_found" if success
                                              else "search_exhausted"))
        all_msgs = []
        total_tokens = 0
        total_latency = 0
        all_tools = []
        for lr in loops:
            all_msgs.extend(lr.messages)
            total_tokens += lr.total_tokens
            total_latency += lr.total_latency_ms
            all_tools.extend(lr.tools_called)
        return LoopResult(
            content=loops[-1].content,
            proof_code=proof_code,
            messages=all_msgs,
            turns_used=sum(lr.turns_used for lr in loops),
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            tools_called=all_tools,
            stopped_reason=("proof_found" if success else "search_exhausted"),
        )

    async def _build_briefing(self, problem) -> str:
        try:
            from knowledge.reader import KnowledgeReader
            reader = KnowledgeReader(self.knowledge_store)
            return await reader.render_for_prompt(
                theorem=problem.theorem_statement,
                max_chars=1500)
        except Exception as e:
            logger.debug(f"briefing skipped: {e}")
            return ""

    # ──────────────────────────────────────────────────────────────────
    # Initial prompt assembly + post-loop auto-verify
    # ──────────────────────────────────────────────────────────────────

    def _build_initial_message(self, problem, profile: Profile) -> str:
        """构造富初始 user message: 题目 + 检索引理 + few-shot。

        v2 之前只有"Prove the theorem"一行, 实际上等于让 LLM 在零上下文下盲做。
        v3 起按 ``profile.observation`` 注入:
          - inject_premises_in_prompt: top-N 检索引理 (供 whole_proof 等无 premise_search 工具的 profile)
          - inject_few_shot: few-shot 示例 (DeepSeek-Prover/Goedel 风格)

        Step-level profile (reprover/leandojo) 默认已经有 premise_search 工具,
        通常 inject_premises_in_prompt 仍开但 n 较少, 让 LLM 主动检索。
        """
        parts = [
            "## Theorem to prove",
            f"```lean\n{problem.theorem_statement}\n```",
        ]

        nl = getattr(problem, "natural_language", "") or ""
        if nl:
            parts.append(f"\n## Informal statement\n{nl}")

        # Few-shot
        if profile.observation.inject_few_shot:
            try:
                from common.prompt_builder import FEW_SHOT_EXAMPLES
                parts.append(f"\n{FEW_SHOT_EXAMPLES}")
            except Exception as e:
                logger.debug(f"few-shot skipped: {e}")

        # Retrieved premises
        if profile.observation.inject_premises_in_prompt and self.retriever:
            premises = self._fetch_premises(
                problem, top_k=profile.observation.n_premises)
            if premises:
                parts.append("\n## Potentially useful Mathlib lemmas")
                for p in premises:
                    parts.append(f"- `{p}`")

        # Closing directive
        if profile.tools:
            tool_list = ", ".join(
                f"`{t.value}`" for t in profile.tools)
            parts.append(
                f"\n## Task\nProve the theorem. Available tools: {tool_list}. "
                f"Iterate using tool feedback. Output the final proof in a "
                f"single ```lean block. Do NOT use `sorry`."
            )
        else:
            parts.append(
                "\n## Task\nGenerate a complete Lean 4 proof. Output ONLY the "
                "proof body inside a single ```lean block. Do NOT use `sorry`."
            )
        return "\n".join(parts)

    def _fetch_premises(self, problem, top_k: int = 10) -> list[str]:
        if not self.retriever:
            return []
        try:
            results = self.retriever.retrieve(
                problem.theorem_statement, top_k=top_k)
            if not results:
                return []
            if isinstance(results[0], str):
                return list(results)
            return [r.get("statement", r.get("name", ""))
                    for r in results if isinstance(r, dict)]
        except Exception as e:
            logger.debug(f"premise fetch failed: {e}")
            return []

    async def _auto_verify_proof(self, problem, proof_code: str):
        """对 LLM 输出但未主动验证的 proof 跑一次 Lean4 编译。

        返回 None 表示无验证器可用; True/False 表示验证结果。
        """
        if not self.lean_pool:
            return None
        try:
            full_code = f"{problem.theorem_statement}\n{proof_code}"
            r = self.lean_pool.check_proof(full_code, timeout=30)
            ok = r.get("success", False) and not r.get("errors")
            return bool(ok)
        except Exception as e:
            logger.debug(f"auto-verify failed: {e}")
            return None

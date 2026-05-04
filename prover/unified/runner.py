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
    # v3.0: full search-tree payload for tree-search profiles. None for
    # linear / parallel runs; rendered into ``meta.search_tree`` in dialog.json.
    search_tree: Optional[dict] = None
    # v6: structured backend-status block. Populated by the runner before
    # return; rendered into ``meta.backends`` by ``save_unified``. Tells
    # downstream consumers which backend actually serviced the run vs
    # which was *requested* — silently degraded "is_fallback=True"
    # configurations are now first-class data instead of a debug-log line.
    # Shape::
    #   {"kimina":     {"present": bool, "is_fallback": bool, ...},
    #    "pantograph": {"present": bool, "is_fallback": bool, "mode": str},
    #    "lookeng":    {"present": bool, "is_fallback": bool},
    #    "lean_pool":  {"present": bool, "kind": str}}
    # An empty dict means "no backend introspection data captured" —
    # legacy callers / direct UnifiedResult construction without the
    # runner.
    backends_status: dict = field(default_factory=dict)

    def save_unified(self, task_dir: str, *, problem_id: str = "",
                     model: str = "", provider: str = "",
                     system_prompt: str = "",
                     tools: list = None,
                     initial_task: str = ""):
        """Save dialog.json — standard project format.

        For tree-search runs (mcts / best_first / beam), the search tree
        is attached to ``meta.search_tree`` *in addition to* the linear
        ``messages`` list (which carries the solved or best-explored path).
        Linear / parallel runs save unchanged from v2.0 behaviour.

        v6: Backend status (which backend was used, whether it was a
        silent fallback) attaches to ``meta.backends``. Absent when
        ``backends_status`` is empty (e.g. legacy direct construction).
        """
        if self.loop_result is None:
            logger.warning("No loop_result to save")
            return
        # Build the dialog through to_dialog so we can post-attach the tree.
        dialog = self.loop_result.to_dialog(
            problem_id=problem_id, model=model, provider=provider,
            system_prompt=system_prompt, tools=tools,
            initial_task=initial_task,
        )
        meta = dialog.setdefault("meta", {})
        if self.search_tree is not None:
            meta["search_tree"] = self.search_tree
        if self.backends_status:
            meta["backends"] = self.backends_status
        from agent.persistence.unified_storage import save_task
        return save_task(task_dir, dialog)


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

class UnifiedProofRunner:
    """One Profile in, one dialog.json out."""

    def __init__(self, *, llm, lean_pool=None,
                  knowledge_store=None,
                  knowledge_writer=None,
                  world_model=None,
                  retriever=None,
                  broadcast_bus=None,
                  kimina_backend=None,
                  pantograph_backend=None,
                  lookeng_backend=None,
                  dialog_index=None,
                  plugin_loader=None,
                  persistent_lemma_bank=None,
                  policy_engine=None,
                  auto_register_llm_autoformalizer: bool = True):
        """
        v14 新增三个可选注入:
          plugin_loader        prover.plugins.PluginLoader 实例 — 按定理领域
                               注入 few-shot/premises/strategic_hint 到首条
                               user message。预留接口 A 的领域维度。
          persistent_lemma_bank prover.lemma_bank.PersistentLemmaBank 实例 —
                               跨问题/跨会话的 SQLite + BM25 引理库, 接到
                               LemmaBankTool 读 + ConjectureProposeTool 写。
                               预留接口 A 的 lemma 维度。
          policy_engine         engine.policy.PolicyEngine 实例 — 接到
                               AgentLoop, 多轮失败后 declarative early termination。
        三者均默认 None, 不传则保持 v13 行为。
        """
        self.llm = llm
        self.lean_pool = lean_pool
        self.knowledge_store = knowledge_store
        # v4: optional KnowledgeWriter — feeds Layer 1 from every tactic
        # application made by step-level profiles. Defaults to
        # knowledge_store.writer when the store exposes one.
        if knowledge_writer is None and knowledge_store is not None:
            knowledge_writer = getattr(knowledge_store, "writer", None)
        self.knowledge_writer = knowledge_writer
        # v4: optional WorldModel — short-circuits high-confidence
        # tactic-failure predictions before the Lean call. None disables
        # the gate. Use ``engine.world_model.make_world_model(path)`` to
        # build the right impl (Trained if .pkl exists, Mock otherwise).
        self.world_model = world_model
        self.retriever = retriever
        self.broadcast_bus = broadcast_bus
        # Optional infrastructure backends — when present, the matching
        # ToolKit (BATCH_VERIFY / MVAR_FOCUS / DRAFT_HOLE / LEMMA_BY_LEMMA)
        # gets a wired-up tool; when absent, those tools register in
        # fallback mode and return a structured "unavailable" error.
        self.kimina_backend = kimina_backend
        self.pantograph_backend = pantograph_backend
        self.lookeng_backend = lookeng_backend
        # v5: optional DialogIndex for cross-problem demonstration
        # injection. When present and the active profile has
        # ``observation.inject_similar_dialogs=True``, similar past
        # solved dialogs get prepended to the initial user message
        # as in-context demos. None disables the feature.
        self.dialog_index = dialog_index
        # v14 (项④): optional plugin loader for domain-specific injection.
        self.plugin_loader = plugin_loader
        # v14 (项③): optional persistent lemma bank — cross-problem reuse.
        self.persistent_lemma_bank = persistent_lemma_bank
        # v14 (项②): optional policy engine — declarative early-term rules.
        self.policy_engine = policy_engine

        # v6: by default, if the runner has an LLM and no autoformalizer
        # has been registered yet, plug the LLM in as the default NL→FL
        # translator for ``NLExistenceBridgeTool``. This closes the
        # "5-pattern heuristic is silly when an LLM is on the bench"
        # gap by making the LLM path the *default* instead of opt-in.
        #
        # Three guardrails:
        #
        #   1. We never overwrite an explicit prior registration. If
        #      the user already called ``register_autoformalizer(...)``
        #      or ``register_llm_autoformalizer(...)`` before
        #      constructing the runner, we leave that callable in place.
        #   2. Pass ``auto_register_llm_autoformalizer=False`` to
        #      preserve the V5-and-earlier behaviour (heuristic until
        #      the user explicitly registers).
        #   3. Registration failures are swallowed — autoformalization
        #      is a best-effort path, not a precondition for proof.
        self._auto_registered_autoformalizer = False
        if auto_register_llm_autoformalizer and llm is not None:
            self._maybe_auto_register_autoformalizer()

    def _maybe_auto_register_autoformalizer(self) -> None:
        """Register the LLM as the default NL→FL translator iff none set.

        Safe to call multiple times — only the first call (in any
        process) actually registers. Subsequent runners observe a
        non-None registry and leave it alone, so two runners
        constructed back-to-back don't fight over the registration.

        Behaviour:
          * already-set registry → no-op
          * import failure (tools_infra/llm_autoformalizer) → log
            debug, no-op (the heuristic path still works)
          * llm doesn't expose ``.generate`` or ``.agenerate`` →
            no-op (autoformalizer factory would reject it anyway)
          * happy path → register, set
            ``_auto_registered_autoformalizer = True``

        We deliberately do not touch the heuristic registration path:
        the autoformalizer registry is module-level, so a single
        successful registration affects every NLExistenceBridgeTool
        in the process — but the heuristic remains the documented
        fallback when the registered callable raises.
        """
        try:
            from prover.unified.tools_infra import (
                _get_autoformalizer, register_autoformalizer)
            from prover.unified.llm_autoformalizer import (
                make_llm_autoformalizer)
        except ImportError as e:
            logger.debug(
                f"[unified] auto-register autoformalizer skipped: {e}")
            return

        if _get_autoformalizer() is not None:
            # Caller already set up an autoformalizer (LLM or otherwise).
            # Do not clobber.
            return

        # ``make_llm_autoformalizer`` validates the LLM shape and
        # raises if .generate is missing — wrap the whole thing in a
        # broad except so a non-conforming LLM doesn't break runner
        # construction.
        try:
            fn = make_llm_autoformalizer(self.llm)
            register_autoformalizer(fn)
            self._auto_registered_autoformalizer = True
            logger.debug(
                "[unified] auto-registered LLM autoformalizer "
                "(default NL→FL translator now uses runner.llm)")
        except Exception as e:  # noqa: BLE001
            logger.debug(
                f"[unified] LLM autoformalizer registration failed: {e}")

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
        # v6: capture backend introspection so the dialog records which
        # backend actually serviced the run (vs which was requested).
        # Done after dispatch so the data reflects post-run state
        # (e.g. a backend that started healthy but degraded mid-run).
        ur.backends_status = self._collect_backend_status()
        return ur

    def _collect_backend_status(self) -> dict:
        """Snapshot the four backend slots into a structured dict.

        Returns a dict whose top-level keys are the four backend slots
        the runner knows about (``kimina``, ``pantograph``, ``lookeng``,
        ``lean_pool``). Each value is a sub-dict with at least:

          * ``present`` — whether the runner has a backend object in
            this slot at all
          * ``is_fallback`` — for community backends, whether the
            backend silently degraded (e.g. pypantograph not installed,
            Kimina server unreachable). Always ``False`` for ``lean_pool``
            since the local Lean pool has no fallback concept.

        Plus per-backend extras (``mode`` for pantograph, ``kind`` for
        lean_pool, etc.) when the backend exposes them. Missing
        attributes are tolerated — older or third-party backend
        implementations that don't expose ``is_fallback`` get
        ``"is_fallback": None`` so callers can distinguish "definitely
        not fallback" (False) from "we don't know" (None).

        Returns ``{}`` (empty dict) only on a complete introspection
        failure — never raises. The contract is "best-effort, fail-soft":
        the dialog still writes even if introspection blows up.
        """
        status: dict = {}

        def _peek(slot_name: str, backend, *,
                   want_attrs: tuple = ("is_fallback",),
                   want_method: tuple = ()) -> dict:
            """Read attributes from a backend object, tolerating any failure."""
            entry: dict = {"present": backend is not None}
            if backend is None:
                return entry
            for attr in want_attrs:
                try:
                    val = getattr(backend, attr, None)
                    # Properties may compute things; coerce to plain bool/str
                    if val is True or val is False:
                        entry[attr] = bool(val)
                    elif val is None:
                        entry[attr] = None
                    else:
                        entry[attr] = str(val)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        f"_collect_backend_status: {slot_name}.{attr} "
                        f"raised {type(e).__name__}: {e}")
                    entry[attr] = None
            for meth in want_method:
                try:
                    fn = getattr(backend, meth, None)
                    if callable(fn):
                        entry[meth] = fn()
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        f"_collect_backend_status: {slot_name}.{meth}() "
                        f"raised {type(e).__name__}: {e}")
            return entry

        try:
            status["kimina"] = _peek(
                "kimina", self.kimina_backend,
                want_attrs=("is_fallback", "is_alive"))
            status["pantograph"] = _peek(
                "pantograph", self.pantograph_backend,
                want_attrs=("is_fallback", "mode"))
            status["lookeng"] = _peek(
                "lookeng", self.lookeng_backend,
                want_attrs=("is_fallback",))
            # lean_pool has a different shape — it's the local
            # AsyncLeanPool, not a community REPLTransport. No
            # fallback concept; we record only its class name as
            # ``kind`` so the dialog shows whether the run used the
            # real pool, a Mock pool, or a custom pool.
            lean_entry: dict = {"present": self.lean_pool is not None}
            if self.lean_pool is not None:
                lean_entry["kind"] = type(self.lean_pool).__name__
                # Optional: many pool implementations expose ``size``
                # or ``alive`` — record if present, ignore if not.
                for opt in ("size", "is_alive"):
                    val = getattr(self.lean_pool, opt, None)
                    if val is True or val is False:
                        lean_entry[opt] = bool(val)
                    elif isinstance(val, int):
                        lean_entry[opt] = val
            status["lean_pool"] = lean_entry
        except Exception as e:  # noqa: BLE001
            # Hard fail-soft: any unexpected failure during
            # introspection yields an empty dict. The dialog still
            # saves; downstream consumers see no backends block and
            # treat that as "no introspection data".
            logger.warning(
                f"[unified] backend status collection failed: {e}")
            return {}
        return status

    # ──────────────────────────────────────────────────────────────────
    # Mode A: single AgentLoop, no outer search
    # ──────────────────────────────────────────────────────────────────

    async def _run_single_loop(self, problem, profile: Profile) -> UnifiedResult:
        """Whole-proof / repair / DSP / ReProver / LeanDojo —— 都走这条."""
        registry = build_tool_registry(
            profile,
            lean_pool=self.lean_pool,
            knowledge_store=self.knowledge_store,
            knowledge_writer=self.knowledge_writer,
            world_model=self.world_model,
            retriever=self.retriever,
            broadcast_bus=self.broadcast_bus,
            search_state=None,
            kimina_backend=self.kimina_backend,
            pantograph_backend=self.pantograph_backend,
            lookeng_backend=self.lookeng_backend,
            persistent_lemma_bank=self.persistent_lemma_bank,
            llm=self.llm,
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

        # ── LooKeng: pre-bootstrap a session so the LLM never has to
        # invent a session_id. The id is threaded through ToolContext.
        # The bootstrap is best-effort: if the backend is unavailable
        # we leave shared_state empty and the LemmaByLemmaTool will
        # report a structured error on the LLM's first call.
        if self.lookeng_backend is not None and any(
                t == ToolKit.LEMMA_BY_LEMMA for t in profile.tools):
            try:
                sid = await self.lookeng_backend.begin_session(
                    theorem=problem.theorem_statement)
                tool_ctx.shared_state["lookeng_session_id"] = sid
                logger.info(
                    f"[unified] LooKeng session pre-bootstrapped: {sid}")
            except Exception as e:
                logger.warning(
                    f"LooKeng begin_session failed (will retry on first "
                    f"tool call): {e}")

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
            knowledge_writer=self.knowledge_writer,
            world_model=self.world_model,
            retriever=self.retriever,
            broadcast_bus=self.broadcast_bus,
            search_state=state,        # ← 关键: tools 持有同一 state
            kimina_backend=self.kimina_backend,
            pantograph_backend=self.pantograph_backend,
            lookeng_backend=self.lookeng_backend,
            persistent_lemma_bank=self.persistent_lemma_bank,
            llm=self.llm,
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
            # v3.0: stash this expansion's messages on the node so the
            # final dialog.json can reproduce the search tree faithfully.
            try:
                msgs_dicts = self._loop_messages_to_dicts(lr)
            except Exception as e:
                logger.debug(f"loop→dict conversion failed: {e}")
                msgs_dicts = []
            target_node = state.nodes.get(node_id)
            if target_node is not None:
                # Any new children created during this expansion belong
                # to *this* expansion's transcript; record on the parent
                # since they share the same LLM turn(s).
                target_node.messages.extend(msgs_dicts)

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

        # Build the linear "best path" view into a LoopResult; the full
        # tree rides separately on UnifiedResult.search_tree and lands
        # under meta.search_tree at save time.
        merged = self._merge_loops_with_tree(
            state, profile, all_loop_results, proof_code, success)
        tree_dict = state.to_search_tree_dict(kind=profile.search.kind)

        return UnifiedResult(
            profile_name=profile.name,
            success=success,
            proof_code=proof_code,
            loop_result=merged,
            search_tree=tree_dict,
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
        """异构 N 个 sub-profile 并行 + 共享广播总线 (项目原有特色).

        v13: 真正接通 broadcast bus —— 之前 sub-profile 用 ``sp.__dict__``
        实例化, parent profile 的 ``ToolKit.BROADCAST`` 不传播, 整个
        ``engine/broadcast.py`` 在主路径死代码, heterogeneous 实际只是
        best-of-4。现把 BROADCAST 注入每个 sub-profile 的 tools 列表,
        并把 bus 透传给所有 sub-runners (它们走同一个 ``self``, 共享
        ``self.broadcast_bus``)。一个 sub-profile 提议「avoid: ring 在 ℕ
        减法上无效」, 其他三个 sub-profile 立刻能 read 到。
        """
        sub_names = profile.search.parallel_profiles or [profile.name]
        sub_profiles = [get_profile(n) for n in sub_names]
        # Share broadcast bus across all sub-runs
        if self.broadcast_bus is None:
            try:
                from engine.broadcast import BroadcastBus
                self.broadcast_bus = BroadcastBus()
            except Exception:
                self.broadcast_bus = None

        def _augment(sp: Profile) -> Profile:
            """Clone sub-profile with BROADCAST tool merged in + search reset."""
            new_tools = list(sp.tools)
            if (self.broadcast_bus is not None
                    and ToolKit.BROADCAST not in new_tools):
                new_tools.append(ToolKit.BROADCAST)
            return sp.__class__(**{
                **sp.__dict__,
                "tools": new_tools,
                # Reset search to none for sub-profiles to avoid recursion.
                "search": SearchConfig(kind="none"),
            })

        tasks = [self.run(problem, profile=_augment(sp)) for sp in sub_profiles]
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
                            "sub_profiles": sub_names,
                            "broadcast_bus": self.broadcast_bus is not None},
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
        # v14 (项②): inject policy_engine if available, else stay silent.
        return AgentLoop(
            llm=self.llm, tools=registry, config=config,
            policy_engine=self.policy_engine)

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

    # ── v3.0: tree-aware merge — only the solved path lands in `messages`,
    # the rest of the tree rides under meta.search_tree. ─────────────────

    def _loop_messages_to_dicts(self, lr: LoopResult) -> list[dict]:
        """Convert a LoopResult's messages (LoopMessage objects) into
        plain dialog message dicts for storage on a TreeNode."""
        out: list[dict] = []
        for m in (lr.messages or []):
            # LoopMessage might already be dict-like; tolerate both.
            if isinstance(m, dict):
                out.append(dict(m))
                continue
            d: dict = {"role": getattr(m, "role", "assistant")}
            content = getattr(m, "content", "")
            if content:
                d["content"] = content
            thought = getattr(m, "thought", None)
            if thought:
                d["thought"] = thought
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                d["tool_calls"] = [
                    tc if isinstance(tc, dict) else (
                        tc.to_dict() if hasattr(tc, "to_dict")
                        else {"id": getattr(tc, "id", ""),
                              "function": {
                                "name": getattr(tc, "name", ""),
                                "arguments": getattr(tc, "arguments", "")},
                              "server_id": getattr(tc, "server_id",
                                                    "default")})
                    for tc in tcs
                ]
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                d["tool_call_id"] = tcid
            name = getattr(m, "name", None)
            if name:
                d["name"] = name
            sid = getattr(m, "server_id", None)
            if sid:
                d["server_id"] = sid
            out.append(d)
        return out

    def _merge_loops_with_tree(self, state, profile,
                                  loops: list[LoopResult],
                                  proof_code: str,
                                  success: bool) -> LoopResult:
        """For tree-search profiles, the linear ``messages`` list holds
        the solved-or-best path only. Aggregate stats across all loops.

        Compare with ``_merge_loops`` (used by `parallel`): there we
        concatenate every sub-loop's messages because each is a real
        independent attempt; here we don't, because a sibling branch
        is *not* a path the agent committed to."""
        if not loops:
            return LoopResult(
                content="",
                proof_code=proof_code,
                stopped_reason=("proof_found" if success
                                  else "search_exhausted"),
            )

        # Linear messages = solved-path messages from the tree state.
        solved_path = state.solved_path_messages()

        # Aggregate stats across every expansion.
        total_tokens = sum(lr.total_tokens for lr in loops)
        total_latency = sum(lr.total_latency_ms for lr in loops)
        all_tools: list = []
        for lr in loops:
            all_tools.extend(lr.tools_called or [])

        return LoopResult(
            content=loops[-1].content,
            proof_code=proof_code,
            messages=solved_path,
            turns_used=sum(lr.turns_used for lr in loops),
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            tools_called=all_tools,
            stopped_reason=("proof_found" if success
                            else "search_exhausted"),
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
                from common.few_shot import FEW_SHOT_EXAMPLES
                parts.append(f"\n{FEW_SHOT_EXAMPLES}")
            except Exception as e:
                logger.debug(f"few-shot skipped: {e}")

        # v14: domain plugin injection (prover/plugins/) — 按定理领域注入
        # 额外 few-shot + premises + strategic hint。运行时 lazy-load, 找不到
        # 插件目录或没有匹配则静默跳过, 与 v13 行为一致。
        if self.plugin_loader is not None:
            try:
                matches = self.plugin_loader.match(problem.theorem_statement)
                if matches:
                    top = matches[0]  # 取得分最高的一个插件
                    if top.few_shot_examples:
                        parts.append(f"\n## Domain-specific examples ({top.name})")
                        parts.append(top.few_shot_examples)
                    if top.extra_premises:
                        parts.append(f"\n## Domain-specific lemmas ({top.name})")
                        for prem in top.extra_premises[:10]:
                            stmt = prem.get("statement", "")
                            if stmt:
                                parts.append(f"- `{stmt}`")
                    if top.strategic_hint:
                        parts.append(f"\n## Strategic hint\n{top.strategic_hint}")
            except Exception as e:
                logger.debug(f"plugin injection skipped: {e}")

        # v5: similar past dialogs (cross-problem demo retrieval)
        if profile.observation.inject_similar_dialogs \
                and self.dialog_index is not None:
            try:
                similar_block = self.dialog_index.render_for_prompt(
                    problem.theorem_statement,
                    top_k=profile.observation.n_similar_dialogs,
                    max_chars=profile.observation.similar_dialogs_max_chars,
                    solved_only=True)
                if similar_block:
                    parts.append("\n" + similar_block.rstrip())
            except Exception as e:
                logger.debug(f"similar-dialog injection skipped: {e}")

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

        v10: 修了 v9 留下的 method-name bug。pool 的真实接口是
        ``verify_complete(theorem, proof, preamble)`` ——
        ``check_proof`` 从来不存在,旧代码每次都会落到 except 分支,
        导致 auto_inject_lean_compile 兜底实际从未生效。
        """
        if not self.lean_pool:
            return None
        try:
            verify = getattr(self.lean_pool, "verify_complete", None)
            if verify is None:
                logger.debug(
                    f"auto-verify: pool {type(self.lean_pool).__name__} "
                    f"has no verify_complete")
                return None
            import inspect as _inspect
            result = verify(problem.theorem_statement, proof_code, "")
            if _inspect.iscoroutine(result):
                result = await result
            success = bool(getattr(result, "success", False))
            has_sorry = bool(getattr(result, "has_sorry", False))
            errors = getattr(result, "errors", None) or []
            return bool(success and not has_sorry and not errors)
        except Exception as e:
            logger.debug(f"auto-verify failed: {e}")
            return None

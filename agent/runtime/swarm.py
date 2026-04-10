"""agent/runtime/swarm.py — Swarm coordinator for multi-agent proof exploration

Inspired by Claude Code's swarm system (utils/swarm/):
  - A lead agent decomposes the problem
  - Worker agents tackle sub-goals in parallel
  - Results flow back through the mailbox
  - The lead agent fuses partial results into a complete proof

This replaces the flat AgentPool with a hierarchical coordination model
where agents can dynamically spawn, communicate, and compose results.

Usage::

    swarm = ProofSwarm(
        llm=provider,
        tool_registry=registry,
        config=SwarmConfig(max_workers=4),
    )
    result = await swarm.solve(problem)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from agent.runtime.sub_agent import (
    AgentSpec, AgentTask, AgentResult, ContextItem,
)
from agent.runtime.mailbox import AgentMailbox, AgentMessage, MessageTopic
from agent.runtime.agent_tool import SpawnAgentTool, DEFAULT_AGENT_TYPES
from agent.runtime.result_fuser import ResultFuser
from agent.tools.base import ToolContext
from agent.tools.registry import ToolRegistry
from common.roles import AgentRole, ROLE_PROMPTS
from common.budget import Budget

logger = logging.getLogger(__name__)


@dataclass
class SwarmConfig:
    """Configuration for the proof swarm."""
    max_workers: int = 4             # Max concurrent sub-agents
    max_rounds: int = 3              # Max lead-agent reasoning rounds
    lead_max_turns: int = 15         # Max agentic loop turns for lead
    worker_max_turns: int = 8        # Max turns per worker
    timeout_seconds: float = 300.0   # Total swarm timeout
    max_total_tokens: int = 500_000  # Total token budget across all agents
    # Strategy
    enable_decomposition: bool = True   # Lead agent can decompose into sub-goals
    enable_cross_pollination: bool = True  # Workers share discoveries via mailbox
    early_stop_on_proof: bool = True    # Stop all workers when one succeeds


@dataclass
class SwarmResult:
    """Result from the swarm proof attempt."""
    success: bool = False
    proof_code: str = ""
    best_result: Optional[AgentResult] = None
    all_results: list[AgentResult] = field(default_factory=list)
    rounds_used: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    workers_spawned: int = 0
    stop_reason: str = ""
    discoveries: list[dict] = field(default_factory=list)  # Shared findings


LEAD_AGENT_PROMPT = """\
You are the lead mathematician coordinating a team of specialist agents to \
prove a Lean 4 theorem. Your role:

1. ANALYZE the theorem to identify its mathematical domain and difficulty.
2. PLAN a proof strategy: direct proof, induction, contradiction, decomposition.
3. DELEGATE sub-tasks to specialist agents using the spawn_agent tool.
4. SYNTHESIZE results from sub-agents into a complete proof.
5. If a sub-agent's proof is close but has errors, spawn a repair_specialist.

Available specialist agents:
- induction_expert: For inductive proofs on Nat, List, recursive structures
- algebra_expert: For ring/field identities, commutativity, algebraic rewrites
- repair_specialist: Fix broken proofs given errors
- decomposer: Break complex goals into sub-lemmas
- tactic_explorer: Try many tactics quickly for automatable goals
- critic: Review and evaluate proof attempts

You can also use tools directly: premise_search, goal_inspect, lean_verify.

When you have a complete proof, output it in a ```lean block.
NEVER use sorry or admit in your final proof.
"""


class ProofSwarm:
    """Hierarchical multi-agent proof coordinator.

    Architecture:
        Lead Agent (with SpawnAgentTool)
            ├── Worker 1 (induction_expert)
            ├── Worker 2 (algebra_expert)
            ├── Worker 3 (tactic_explorer)
            └── ...
        All connected via AgentMailbox for cross-pollination.
    """

    def __init__(
        self,
        llm: Any,  # AsyncLLMProvider or LLMProvider
        tool_registry: ToolRegistry = None,
        config: SwarmConfig = None,
    ):
        self.llm = llm
        self.base_registry = tool_registry or ToolRegistry()
        self.config = config or SwarmConfig()
        self.mailbox = AgentMailbox()
        self.fuser = ResultFuser()

    async def solve(
        self,
        theorem_statement: str,
        problem_context: str = "",
        knowledge_context: str = "",
        budget: Budget = None,
    ) -> SwarmResult:
        """Run the swarm to prove a theorem.

        Args:
            theorem_statement: Full Lean 4 theorem statement
            problem_context: Additional context (imports, definitions)
            knowledge_context: Knowledge from knowledge store
            budget: Shared token budget

        Returns:
            SwarmResult with best proof and all agent results
        """
        start = time.time()
        budget = budget or Budget(max_tokens=self.config.max_total_tokens)

        # Build tool registry for lead agent (includes SpawnAgentTool)
        lead_registry = self._build_lead_registry()

        # Build lead agent spec
        lead_spec = AgentSpec(
            name="lead_agent",
            role=AgentRole.PROOF_PLANNER,
            temperature=0.5,
            tools=lead_registry.list_tools(),
            context_budget=100_000,
            timeout_seconds=self.config.timeout_seconds,
            system_prompt_override=LEAD_AGENT_PROMPT,
        )

        # Build task
        task_parts = [f"Prove the following Lean 4 theorem:\n\n{theorem_statement}"]
        if problem_context:
            task_parts.append(f"\nContext:\n{problem_context}")
        if knowledge_context:
            task_parts.append(f"\nRelevant knowledge:\n{knowledge_context}")

        task = AgentTask(
            description="\n".join(task_parts),
            theorem_statement=theorem_statement,
        )

        # Run lead agent with agentic loop
        swarm_result = SwarmResult()

        try:
            from agent.runtime.sub_agent import SubAgent

            lead = SubAgent(lead_spec, self.llm, lead_registry)
            result = await lead.execute_with_tools(
                task, lead_registry, max_turns=self.config.lead_max_turns)

            swarm_result.all_results.append(result)
            swarm_result.total_tokens += result.tokens_used
            swarm_result.total_latency_ms = int((time.time() - start) * 1000)
            swarm_result.rounds_used = 1

            if result.proof_code and "sorry" not in result.proof_code:
                swarm_result.success = True
                swarm_result.proof_code = result.proof_code
                swarm_result.best_result = result
                swarm_result.stop_reason = "proof_found"
            else:
                swarm_result.stop_reason = "no_proof"
                swarm_result.best_result = result

        except Exception as e:
            logger.error(f"Swarm lead agent failed: {e}")
            swarm_result.stop_reason = f"error: {e}"

        # Collect discoveries from mailbox
        all_msgs = self.mailbox.receive("*", max_messages=50, mark_delivered=False)
        swarm_result.discoveries = [
            {"from": m.from_agent, "topic": m.topic, "content": m.content[:200]}
            for m in all_msgs
        ]

        return swarm_result

    def _build_lead_registry(self) -> ToolRegistry:
        """Build tool registry for the lead agent.

        Includes all base tools plus SpawnAgentTool.
        """
        registry = ToolRegistry()

        # Copy base tools
        for name in self.base_registry.list_tools():
            tool = self.base_registry.get(name)
            if tool:
                registry.register(tool)

        # Add SpawnAgentTool
        spawn_tool = SpawnAgentTool(
            llm_provider=self.llm,
            tool_registry=self.base_registry,  # Workers get base tools only
            agent_types=DEFAULT_AGENT_TYPES,
            mailbox=self.mailbox,
            max_sub_turns=self.config.worker_max_turns,
        )
        registry.register(spawn_tool)

        return registry

    async def solve_parallel(
        self,
        theorem_statement: str,
        directions: list[dict] = None,
        budget: Budget = None,
    ) -> SwarmResult:
        """Run multiple independent proof attempts in parallel.

        Simpler than the full hierarchical solve(): launches N workers
        with different strategies and returns the best result.

        Args:
            theorem_statement: Full theorem statement
            directions: List of {"role": ..., "temperature": ..., "hint": ...}
            budget: Shared budget

        Returns:
            SwarmResult with best proof from any worker
        """
        start = time.time()
        budget = budget or Budget(max_tokens=self.config.max_total_tokens)

        if not directions:
            directions = [
                {"role": AgentRole.PROOF_GENERATOR, "temperature": 0.5,
                 "hint": "Try direct proof with automation tactics."},
                {"role": AgentRole.PROOF_GENERATOR, "temperature": 0.8,
                 "hint": "Try induction if applicable, otherwise structural proof."},
                {"role": AgentRole.PROOF_GENERATOR, "temperature": 0.3,
                 "hint": "Try the simplest approach: simp, ring, omega, decide."},
            ]

        # Build tasks
        tasks = []
        for i, d in enumerate(directions[:self.config.max_workers]):
            spec = AgentSpec(
                name=f"worker_{i}",
                role=d.get("role", AgentRole.PROOF_GENERATOR),
                temperature=d.get("temperature", 0.7),
                tools=self.base_registry.list_tools(),
                timeout_seconds=self.config.timeout_seconds / 2,
            )
            hint = d.get("hint", "")
            task = AgentTask(
                description=f"{hint}\n\nProve:\n{theorem_statement}",
                theorem_statement=theorem_statement,
            )
            tasks.append((spec, task))

        # Run in parallel
        from agent.runtime.sub_agent import SubAgent

        async def run_worker(spec, task):
            agent = SubAgent(spec, self.llm, self.base_registry)
            if hasattr(agent, 'execute_with_tools'):
                return await agent.execute_with_tools(
                    task, self.base_registry,
                    max_turns=self.config.worker_max_turns)
            return agent.execute(task)

        coros = [run_worker(s, t) for s, t in tasks]

        if self.config.early_stop_on_proof:
            results = await self._race_for_proof(coros)
        else:
            results = await asyncio.gather(*coros, return_exceptions=True)
            results = [r for r in results if not isinstance(r, Exception)]

        # Find best result
        swarm_result = SwarmResult(
            all_results=results,
            workers_spawned=len(tasks),
            total_latency_ms=int((time.time() - start) * 1000),
        )

        for r in results:
            swarm_result.total_tokens += r.tokens_used
            if (r.proof_code and "sorry" not in r.proof_code
                    and (not swarm_result.best_result
                         or r.confidence > swarm_result.best_result.confidence)):
                swarm_result.best_result = r
                swarm_result.proof_code = r.proof_code
                swarm_result.success = True

        swarm_result.stop_reason = "proof_found" if swarm_result.success else "no_proof"
        return swarm_result

    async def _race_for_proof(
        self,
        coros: list,
    ) -> list[AgentResult]:
        """Race coroutines, cancel others when one finds a proof."""
        tasks = [asyncio.create_task(c) for c in coros]
        results = []
        found_proof = False

        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
                results.append(result)
                if (result.proof_code
                        and "sorry" not in result.proof_code):
                    found_proof = True
                    # Cancel remaining tasks
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except (asyncio.CancelledError, Exception):
                continue

        # Collect any already-completed results
        if not found_proof:
            for t in tasks:
                if t.done() and not t.cancelled():
                    try:
                        results.append(t.result())
                    except Exception as _exc:
                        logger.debug(f"Suppressed exception: {_exc}")

        return results

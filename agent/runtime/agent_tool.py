"""agent/runtime/agent_tool.py — Spawn sub-agents as tool calls

Inspired by Claude Code's AgentTool (tools/AgentTool/AgentTool.tsx):
a lead agent can delegate sub-tasks to specialized sub-agents by
calling this tool, similar to how Claude Code spawns sub-agents with
independent context windows.

This enables hierarchical proof decomposition: the lead agent
decomposes a theorem into sub-goals, then spawns specialist agents
(induction expert, algebra expert, etc.) to tackle each one.

Usage in the tool registry::

    registry.register(SpawnAgentTool(
        agent_pool=pool,
        agent_specs=available_specs,
        mailbox=mailbox,
    ))

Then in the LLM prompt, the agent can call::

    tool_use: spawn_agent
    input: {
        "agent_type": "induction_expert",
        "task": "Prove by induction: n + 0 = n",
        "context": "We know Nat.add_succ and Nat.add_zero..."
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from agent.tools.base import Tool, ToolContext, ToolResult, ToolPermission
from agent.runtime.sub_agent import AgentSpec, AgentTask, AgentResult, ContextItem
from common.roles import AgentRole

logger = logging.getLogger(__name__)


# ── Default agent type definitions ───────────────────────────────────────────

DEFAULT_AGENT_TYPES: dict[str, dict] = {
    "induction_expert": {
        "role": AgentRole.PROOF_GENERATOR,
        "description": "Specialist in inductive proofs. Use for goals involving natural numbers, lists, or recursive structures.",
        "temperature": 0.6,
        "tools": ["premise_search", "goal_inspect", "lean_verify", "tactic_suggest"],
    },
    "algebra_expert": {
        "role": AgentRole.PROOF_GENERATOR,
        "description": "Specialist in algebraic manipulation. Use for ring/field identities, commutativity, associativity.",
        "temperature": 0.5,
        "tools": ["premise_search", "lean_verify", "cas_evaluate"],
    },
    "repair_specialist": {
        "role": AgentRole.REPAIR_AGENT,
        "description": "Fixes broken proofs. Give it a failed proof and error messages.",
        "temperature": 0.7,
        "tools": ["goal_inspect", "lean_verify", "tactic_suggest", "premise_search"],
    },
    "decomposer": {
        "role": AgentRole.DECOMPOSER,
        "description": "Breaks complex theorems into smaller lemmas. Use when the goal is too large for direct proof.",
        "temperature": 0.6,
        "tools": ["premise_search", "goal_inspect"],
    },
    "tactic_explorer": {
        "role": AgentRole.PROOF_GENERATOR,
        "description": "Tries many tactics quickly. Use for goals that might yield to automation.",
        "temperature": 0.9,
        "tools": ["tactic_suggest", "lean_auto", "lean_verify"],
    },
    "critic": {
        "role": AgentRole.CRITIC,
        "description": "Reviews proof attempts and suggests improvements. Use to evaluate partial progress.",
        "temperature": 0.3,
        "tools": ["goal_inspect", "lean_verify"],
    },
}


class SpawnAgentTool(Tool):
    """Tool that allows a lead agent to spawn sub-agents.

    The sub-agent runs with its own context and tool set, then returns
    its result to the lead agent as a tool result.
    """

    name = "spawn_agent"
    description = (
        "Spawn a specialized sub-agent to work on a specific sub-task. "
        "The sub-agent has its own context and can use tools independently. "
        "Available agent types:\n"
    )
    permission = ToolPermission.WRITE_LOCAL
    input_schema = {
        "type": "object",
        "properties": {
            "agent_type": {
                "type": "string",
                "description": (
                    "Type of specialist agent to spawn. Options: "
                    "induction_expert, algebra_expert, repair_specialist, "
                    "decomposer, tactic_explorer, critic"
                ),
            },
            "task": {
                "type": "string",
                "description": "Description of what the sub-agent should do",
            },
            "context": {
                "type": "string",
                "description": "Additional context to inject (relevant lemmas, errors, etc.)",
            },
            "theorem": {
                "type": "string",
                "description": "The theorem statement (if different from main problem)",
            },
        },
        "required": ["agent_type", "task"],
    }

    def __init__(
        self,
        llm_provider=None,
        agent_pool=None,
        tool_registry=None,
        agent_types: dict = None,
        mailbox=None,
        max_sub_turns: int = 8,
    ):
        self._llm = llm_provider
        self._pool = agent_pool
        self._tool_registry = tool_registry
        self._agent_types = agent_types or DEFAULT_AGENT_TYPES
        self._mailbox = mailbox
        self._max_sub_turns = max_sub_turns

        # Update description with available types
        type_lines = []
        for name, info in self._agent_types.items():
            type_lines.append(f"  - {name}: {info.get('description', '')}")
        self.description += "\n".join(type_lines)

    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        agent_type = input["agent_type"]
        task_desc = input["task"]
        extra_context = input.get("context", "")
        theorem = input.get("theorem", "") or ctx.theorem_statement

        # Look up agent type
        type_info = self._agent_types.get(agent_type)
        if not type_info:
            return ToolResult.error(
                f"Unknown agent type: '{agent_type}'. "
                f"Available: {list(self._agent_types.keys())}")

        # Build agent spec
        spec = AgentSpec(
            name=f"sub_{agent_type}_{id(input) % 10000}",
            role=type_info["role"],
            temperature=type_info.get("temperature", 0.7),
            tools=type_info.get("tools", []),
            timeout_seconds=min(60, ctx.budget_remaining_seconds),
        )

        # Build task
        injected = []
        if extra_context:
            injected.append(ContextItem(
                key="lead_context", content=extra_context, priority=0.8))

        # Inject messages from mailbox
        if self._mailbox:
            inbox_text = self._mailbox.format_inbox_for_prompt(spec.name)
            if inbox_text:
                injected.append(ContextItem(
                    key="mailbox", content=inbox_text, priority=0.6))

        task = AgentTask(
            description=task_desc,
            injected_context=injected,
            theorem_statement=theorem,
        )

        # Execute sub-agent
        try:
            from agent.runtime.sub_agent import SubAgent

            if self._llm is None:
                return ToolResult.error("No LLM provider available for sub-agent")

            sub_agent = SubAgent(spec, self._llm, self._tool_registry)

            # Try agentic mode if tool registry available
            if self._tool_registry and hasattr(sub_agent, 'execute_with_tools'):
                result = await sub_agent.execute_with_tools(
                    task, self._tool_registry, max_turns=self._max_sub_turns)
            else:
                result = sub_agent.execute(task)

            # Broadcast findings via mailbox
            if self._mailbox and result.proof_code:
                from agent.runtime.mailbox import MessageTopic
                self._mailbox.broadcast(
                    from_agent=spec.name,
                    topic=MessageTopic.PROOF_FRAGMENT,
                    content=f"Proof fragment from {agent_type}: {result.proof_code[:200]}",
                    data={"proof_code": result.proof_code, "confidence": result.confidence},
                )

            # Format result for the lead agent
            return self._format_sub_result(result, agent_type)

        except Exception as e:
            logger.error(f"Sub-agent {agent_type} failed: {e}")
            return ToolResult.error(f"Sub-agent {agent_type} failed: {e}")

    def _format_sub_result(self, result: AgentResult, agent_type: str) -> ToolResult:
        """Format sub-agent result for the lead agent."""
        parts = [f"Sub-agent [{agent_type}] completed:"]

        if result.error:
            parts.append(f"Error: {result.error}")
            return ToolResult.error("\n".join(parts))

        if result.proof_code:
            parts.append(f"Proof code:\n```lean\n{result.proof_code}\n```")
        else:
            parts.append("No proof code generated.")

        parts.append(f"Confidence: {result.confidence:.2f}")

        meta = result.metadata
        if meta.get("loop_turns"):
            parts.append(f"Reasoning turns: {meta['loop_turns']}")
        if meta.get("tools_called"):
            parts.append(f"Tools used: {meta['tools_called']}")

        if result.content and not result.proof_code:
            # Include analysis/reasoning if no proof
            parts.append(f"Analysis:\n{result.content[:500]}")

        return ToolResult.success("\n".join(parts))

"""agent/tools/base.py — Unified Tool protocol

Inspired by Claude Code's Tool.ts: every tool has a declarative schema,
permission level, input validation, progress callbacks, and timeout handling.

Usage::

    class PremiseSearchTool(Tool):
        name = "premise_search"
        description = "Search Mathlib for relevant lemmas"
        permission = ToolPermission.READ_ONLY
        input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }

        async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
            results = await self.search(input["query"], input.get("max_results", 10))
            return ToolResult.success(json.dumps(results))
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Permission levels (inspired by Claude Code's PermissionMode) ──

class ToolPermission(str, Enum):
    """Tool permission tiers — controls what a tool can do."""
    READ_ONLY = "read_only"        # Read state, no side effects
    WRITE_LOCAL = "write_local"    # Modify proof state, local files
    EXTERNAL = "external"          # Call external services (REPL, CAS, API)
    DANGEROUS = "dangerous"        # Destructive operations (delete, overwrite)


# ── Tool execution context ──

@dataclass
class ToolContext:
    """Context passed to every tool execution.

    Contains the agent's current state, budget, and callback hooks.
    Inspired by Claude Code's ToolUseContext.
    """
    agent_name: str = ""
    theorem_statement: str = ""
    current_goals: list[str] = field(default_factory=list)
    working_dir: str = ""
    budget_remaining_tokens: int = 100_000
    budget_remaining_seconds: float = 120.0
    timeout_seconds: float = 30.0
    # Callbacks
    on_progress: Optional[Callable[[str, float], None]] = None
    # Shared state for cross-tool communication
    shared_state: dict = field(default_factory=dict)
    # Permission mode for the current agent
    allowed_permissions: set[ToolPermission] = field(
        default_factory=lambda: {ToolPermission.READ_ONLY,
                                 ToolPermission.WRITE_LOCAL,
                                 ToolPermission.EXTERNAL})

    def report_progress(self, message: str, fraction: float = -1.0):
        """Report progress to the agent loop."""
        if self.on_progress:
            try:
                self.on_progress(message, fraction)
            except Exception as _exc:
                logger.debug(f"Suppressed exception: {_exc}")


# ── Tool result ──

@dataclass
class ToolResult:
    """Result of a tool execution.

    Inspired by Claude Code's ToolResultBlockParam.
    """
    content: str
    is_error: bool = False
    metadata: dict = field(default_factory=dict)
    latency_ms: int = 0

    @classmethod
    def success(cls, content: str, **metadata) -> ToolResult:
        return cls(content=content, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata) -> ToolResult:
        return cls(content=f"Error: {message}", is_error=True, metadata=metadata)

    def to_message_content(self) -> str:
        """Format for inclusion in LLM conversation."""
        return self.content


# ── Tool base class ──

class Tool(ABC):
    """Base class for all tools.

    Inspired by Claude Code's Tool interface. Every tool declares:
    - name: unique identifier
    - description: for LLM tool-use prompt
    - input_schema: JSON Schema for input validation
    - permission: required permission level
    - execute(): async execution with context
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {}
    permission: ToolPermission = ToolPermission.READ_ONLY

    @abstractmethod
    async def execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        """Execute the tool with validated input.

        Args:
            input: Validated input matching input_schema
            ctx: Execution context with budget, permissions, callbacks

        Returns:
            ToolResult with content string and metadata
        """
        ...

    def validate_input(self, input: dict) -> tuple[bool, str]:
        """Validate input against schema. Returns (valid, error_message)."""
        required = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})

        for key in required:
            if key not in input:
                return False, f"Missing required parameter: '{key}'"

        for key, value in input.items():
            if key in properties:
                prop_schema = properties[key]
                expected_type = prop_schema.get("type")
                if expected_type and not _check_type(value, expected_type):
                    return False, (
                        f"Parameter '{key}': expected {expected_type}, "
                        f"got {type(value).__name__}")

        return True, ""

    def check_permission(self, ctx: ToolContext) -> tuple[bool, str]:
        """Check if the current context allows this tool."""
        if self.permission not in ctx.allowed_permissions:
            return False, (
                f"Tool '{self.name}' requires {self.permission.value} "
                f"permission, not granted")
        return True, ""

    def to_claude_schema(self) -> dict:
        """Convert to Claude API tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    async def safe_execute(self, input: dict, ctx: ToolContext) -> ToolResult:
        """Execute with validation, permission check, timeout, and error handling."""
        # 1. Permission check
        perm_ok, perm_err = self.check_permission(ctx)
        if not perm_ok:
            return ToolResult.error(perm_err)

        # 2. Input validation
        valid, val_err = self.validate_input(input)
        if not valid:
            return ToolResult.error(val_err)

        # 3. Execute with timeout
        start = time.time()
        timeout = min(ctx.timeout_seconds, ctx.budget_remaining_seconds)

        try:
            result = await asyncio.wait_for(
                self.execute(input, ctx), timeout=timeout)
            result.latency_ms = int((time.time() - start) * 1000)
            return result
        except asyncio.TimeoutError:
            elapsed = int((time.time() - start) * 1000)
            return ToolResult.error(
                f"Tool '{self.name}' timed out after {elapsed}ms",
                timeout=True)
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            logger.error(f"Tool '{self.name}' error: {e}", exc_info=True)
            return ToolResult.error(str(e), latency_ms=elapsed)

    def __repr__(self):
        return f"<Tool:{self.name} perm={self.permission.value}>"


def _check_type(value: Any, expected: str) -> bool:
    """Check JSON Schema type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected_type = type_map.get(expected)
    if expected_type is None:
        return True  # unknown type, pass
    return isinstance(value, expected_type)

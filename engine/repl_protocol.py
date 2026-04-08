"""engine/repl_protocol.py — Lean4 REPL wire protocol specification

Lean4 REPL (https://github.com/leanprover-community/repl) uses a JSON-based
protocol over stdin/stdout. This module codifies the exact wire format.

Protocol Summary:
  → Request:  JSON object terminated by newline
  ← Response: JSON object terminated by newline

Request Types:
  1. Command (execute a command in an existing environment):
     {"cmd": "<lean4 code>", "env": <int>}

  2. Tactic (apply a tactic in an existing proof state):
     {"tactic": "<tactic string>", "proofState": <int>}

  3. Command with tactic block (for `by` proofs, gets interactive goals):
     {"cmd": "<theorem := by\\n  sorry>", "env": <int>}

Response Format:
  {
    "env": <int>,                  // New environment ID (for commands)
    "proofState": <int>,           // New proof state ID (for tactics)
    "goals": ["<goal1>", ...],     // Remaining goals (empty = proof complete)
    "messages": [                  // Diagnostics
      {
        "severity": "error"|"warning"|"information",
        "pos": {"line": <int>, "column": <int>},
        "endPos": {"line": <int>, "column": <int>},
        "data": "<message text>"
      }
    ],
    "sorries": [                   // Positions of sorry
      {
        "proofState": <int>,
        "pos": {"line": <int>, "column": <int>},
        "goal": "<goal string>",
        "endPos": {"line": <int>, "column": <int>}
      }
    ]
  }

Key Semantics:
  - env IDs are immutable snapshots: sending {"cmd": "...", "env": 3}
    creates a *new* env (e.g. env=4) without mutating env 3.
  - This enables zero-cost branching: fork from any env_id.
  - proofState IDs similarly allow branching within a proof.
  - An empty goals list after a tactic means the proof is complete.
  - The sorry field provides interactive proof states at sorry positions,
    enabling incremental/interactive proving.

Usage:
    req = REPLRequest.command("import Mathlib", env=0)
    wire = req.to_json()  # '{"cmd": "import Mathlib", "env": 0}'

    resp = REPLResponse.from_json(response_line)
    if resp.has_errors:
        for err in resp.error_messages:
            print(err)
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Request Types ────────────────────────────────────────────

@dataclass
class REPLRequest:
    """A request to the Lean4 REPL."""

    cmd: Optional[str] = None
    tactic: Optional[str] = None
    env: int = 0
    proof_state: Optional[int] = None

    @staticmethod
    def command(code: str, env: int = 0) -> REPLRequest:
        """Execute a Lean4 command (import, definition, theorem, etc.)."""
        return REPLRequest(cmd=code, env=env)

    @staticmethod
    def tactic_step(tactic: str, proof_state: int) -> REPLRequest:
        """Apply a tactic in an existing proof state."""
        return REPLRequest(tactic=tactic, proof_state=proof_state)

    def to_dict(self) -> dict:
        """Serialize to the wire protocol dict."""
        if self.tactic is not None and self.proof_state is not None:
            return {"tactic": self.tactic, "proofState": self.proof_state}
        elif self.cmd is not None:
            return {"cmd": self.cmd, "env": self.env}
        else:
            raise ValueError("REPLRequest must have either cmd or tactic set")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ─── Response Types ───────────────────────────────────────────

@dataclass
class REPLDiagnostic:
    """A single diagnostic message from Lean4."""
    severity: str  # "error", "warning", "information"
    data: str
    pos_line: int = -1
    pos_col: int = -1
    end_line: int = -1
    end_col: int = -1

    @staticmethod
    def from_dict(d: dict) -> REPLDiagnostic:
        pos = d.get("pos", {})
        end = d.get("endPos", {})
        return REPLDiagnostic(
            severity=d.get("severity", "error"),
            data=d.get("data", ""),
            pos_line=pos.get("line", -1),
            pos_col=pos.get("column", -1),
            end_line=end.get("line", -1),
            end_col=end.get("column", -1),
        )

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@dataclass
class REPLSorry:
    """A sorry position with its associated proof state and goal."""
    proof_state: int
    goal: str
    pos_line: int = -1
    pos_col: int = -1
    end_line: int = -1
    end_col: int = -1

    @staticmethod
    def from_dict(d: dict) -> REPLSorry:
        pos = d.get("pos", {})
        end = d.get("endPos", {})
        return REPLSorry(
            proof_state=d.get("proofState", -1),
            goal=d.get("goal", ""),
            pos_line=pos.get("line", -1),
            pos_col=pos.get("column", -1),
            end_line=end.get("line", -1),
            end_col=end.get("column", -1),
        )


@dataclass
class REPLResponse:
    """Parsed response from the Lean4 REPL."""

    raw: dict = field(default_factory=dict)

    # Command response
    env: int = -1

    # Tactic response
    proof_state: Optional[int] = None
    goals: list[str] = field(default_factory=list)

    # Diagnostics
    messages: list[REPLDiagnostic] = field(default_factory=list)

    # Sorry positions (interactive proof states)
    sorries: list[REPLSorry] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> REPLResponse:
        resp = REPLResponse(raw=d)
        resp.env = d.get("env", -1)
        resp.proof_state = d.get("proofState")
        resp.goals = d.get("goals", [])
        resp.messages = [
            REPLDiagnostic.from_dict(m)
            for m in d.get("messages", [])
        ]
        resp.sorries = [
            REPLSorry.from_dict(s)
            for s in d.get("sorries", [])
        ]
        return resp

    @staticmethod
    def from_json(line: str) -> REPLResponse:
        try:
            return REPLResponse.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse REPL response: {e}")
            return REPLResponse(
                messages=[REPLDiagnostic(
                    severity="error",
                    data=f"Invalid REPL response: {e}")])

    @property
    def has_errors(self) -> bool:
        return any(m.is_error for m in self.messages)

    @property
    def error_messages(self) -> list[str]:
        return [m.data for m in self.messages if m.is_error]

    @property
    def is_proof_complete(self) -> bool:
        """True if this was a tactic response with no remaining goals."""
        return len(self.goals) == 0 and not self.has_errors

    @property
    def has_sorry(self) -> bool:
        return len(self.sorries) > 0

    @property
    def interactive_goals(self) -> list[tuple[int, str]]:
        """Return (proof_state_id, goal) pairs from sorries.

        This is the key mechanism for interactive proving:
        send a theorem with `sorry`, get back proof states
        at each sorry position, then use tactic mode to fill them.
        """
        return [(s.proof_state, s.goal) for s in self.sorries]


# ─── Protocol Helpers ─────────────────────────────────────────

def build_sorry_theorem(header: str) -> str:
    """Wrap a theorem header for interactive proving.

    Given: "theorem t (n : Nat) : n + 0 = n"
    Return: "theorem t (n : Nat) : n + 0 = n := by sorry"

    The REPL will return a sorry entry with the proof state
    and goal, which we can then fill interactively via tactic mode.
    """
    header = header.strip()
    if ":=" in header:
        return header  # Already has a body
    return f"{header} := by sorry"


def build_tactic_block(header: str, tactics: list[str]) -> str:
    """Build a complete theorem with a tactic proof block.

    Given: header="theorem t : True", tactics=["trivial"]
    Return: "theorem t : True := by\\n  trivial"
    """
    header = header.strip()
    if ":=" in header:
        # Strip existing body
        header = header[:header.index(":=")].strip()
    body = "\n  ".join(tactics)
    return f"{header} := by\n  {body}"

"""engine/lane/task_packet.py — Structured proof task packet

Inspired by claw-code's TaskPacket (task_packet.rs):
agents receive structured task specifications, not prompt blobs.

Usage::

    packet = ProofTaskPacket(
        theorem_name="nat_add_comm",
        formal_statement="theorem nat_add_comm ...",
        domain="number_theory",
        difficulty="easy",
    )
    validated = validate_packet(packet)  # raises on invalid
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProofTaskPacket:
    """Complete specification for a proof task.

    Analogous to claw-code's TaskPacket — all fields needed to
    independently execute a proof task without ambient context.
    """
    # ── Problem definition ───────────────────────────────────────────
    theorem_name: str
    formal_statement: str
    domain: str = ""
    difficulty: str = "unknown"  # trivial, easy, medium, hard, competition
    natural_language: str = ""   # optional NL description

    # ── Verification policy ──────────────────────────────────────────
    verification_level: str = "l1_repl"  # "l0_only" | "l1_repl" | "l2_full_compile"
    allow_sorry: bool = False
    allow_native_decide: bool = False
    max_verification_timeout_s: int = 30

    # ── Resource budget ──────────────────────────────────────────────
    max_samples: int = 32
    max_api_tokens: int = 500_000
    max_repair_rounds: int = 3
    max_wall_seconds: int = 600

    # ── Strategy configuration ───────────────────────────────────────
    initial_strategy: str = "light"  # "sequential" | "light" | "medium" | "heavy"
    escalation_policy: str = "auto"  # "auto" | "manual" | "none"
    roles: list[str] = field(default_factory=lambda: ["proof_generator", "repair"])
    temperature: float = 0.9
    model_tier: str = "sonnet"  # "haiku" | "sonnet" | "opus"

    # ── Knowledge injection ──────────────────────────────────────────
    inject_knowledge: bool = True
    inject_premises: bool = True
    max_premise_count: int = 10

    # ── Reporting ────────────────────────────────────────────────────
    reporting_level: str = "standard"  # "minimal" | "standard" | "full_trace"
    deposit_knowledge: bool = True
    export_trajectories: bool = False


@dataclass
class PacketValidationError:
    """Validation errors for a task packet."""
    errors: list[str]

    def __str__(self):
        return "; ".join(self.errors)

    def __bool__(self):
        return len(self.errors) > 0


def validate_packet(packet: ProofTaskPacket) -> ProofTaskPacket:
    """Validate a task packet. Raises ValueError on invalid.

    Analogous to claw-code's validate_packet().
    """
    errors = []

    # Required fields
    if not packet.theorem_name.strip():
        errors.append("theorem_name is required")
    if not packet.formal_statement.strip():
        errors.append("formal_statement is required")

    # Value constraints
    if packet.verification_level not in ("l0_only", "l1_repl", "l2_full_compile"):
        errors.append(f"invalid verification_level: {packet.verification_level}")
    if packet.initial_strategy not in ("sequential", "light", "medium", "heavy"):
        errors.append(f"invalid initial_strategy: {packet.initial_strategy}")
    if packet.escalation_policy not in ("auto", "manual", "none"):
        errors.append(f"invalid escalation_policy: {packet.escalation_policy}")
    if packet.model_tier not in ("haiku", "sonnet", "opus"):
        errors.append(f"invalid model_tier: {packet.model_tier}")
    if packet.reporting_level not in ("minimal", "standard", "full_trace"):
        errors.append(f"invalid reporting_level: {packet.reporting_level}")

    # Numeric bounds
    if packet.max_samples < 1:
        errors.append("max_samples must be >= 1")
    if packet.max_wall_seconds < 1:
        errors.append("max_wall_seconds must be >= 1")
    if not (0.0 <= packet.temperature <= 2.0):
        errors.append(f"temperature must be in [0, 2], got {packet.temperature}")

    if errors:
        raise ValueError(f"Invalid task packet: {'; '.join(errors)}")

    return packet


def packet_from_benchmark_problem(problem, config: dict = None) -> ProofTaskPacket:
    """Convert a BenchmarkProblem + config into a validated ProofTaskPacket."""
    config = config or {}
    # BenchmarkProblem uses 'theorem_statement'; fall back to 'formal_statement'
    formal = (getattr(problem, "theorem_statement", "")
              or getattr(problem, "formal_statement", ""))
    return validate_packet(ProofTaskPacket(
        theorem_name=getattr(problem, "name", ""),
        formal_statement=formal,
        domain=getattr(problem, "domain", ""),
        difficulty=getattr(problem, "difficulty", "unknown"),
        natural_language=getattr(problem, "nl_statement", ""),
        max_samples=config.get("max_samples", 32),
        max_api_tokens=config.get("max_api_tokens", 500_000),
        max_wall_seconds=config.get("timeout", 600),
        initial_strategy=config.get("strategy", "light"),
        model_tier=config.get("model_tier", "sonnet"),
        temperature=config.get("temperature", 0.9),
        verification_level=config.get("verification_level", "l1_repl"),
    ))

"""engine/backends/ — Pluggable Lean 4 verification backends.

This package unifies AI4Math with the major Lean 4 infrastructure projects
released by the broader research community. Each backend implements the
``REPLTransport`` ABC from ``engine.transport``, so they all plug into the
existing ``AsyncLeanPool`` / ``UnifiedProofRunner`` without changing a line
of upstream code.

Available backends
------------------

``kimina_server`` — Kimina Lean Server (Numina/Kimi).
    REST-API + multi-process REPL pool + LRU import cache + batch verify
    + infotree-based tactic-state extraction. The reference verifier for
    large-scale RL pipelines (NuminaMath-LEAN, Kimina-Prover-72B).

``pantograph`` — Pantograph (Numina/Stanford/CMU).
    First-class metavariable coupling, drafting (DSP in Lean 4), and
    S-expression proof-term serialization. Adds proof-tree primitives
    that the bare Lean REPL doesn't expose.

``lookeng`` — LooKeng (Seed-Prover 1.5).
    Stateless Python–Lean REPL with a "running context" of proved
    lemmas; the agent inputs one lemma at a time rather than a whole
    proof, giving large I/O reductions on long proofs.

The existing ``LocalTransport`` / ``SocketTransport`` / ``MockTransport``
in ``engine.transport`` remain unchanged and are still the defaults.
The new backends are opt-in via ``--backend kimina|pantograph|lookeng``
on the CLI or via the ``backend`` field in a Profile YAML.
"""
from __future__ import annotations

from engine.backends.kimina_server import (
    KiminaServerBackend,
    KiminaServerClient,
    BatchVerifyRequest,
    BatchVerifyResult,
)
from engine.backends.pantograph import (
    PantographBackend,
    GoalFragment,
    MVarFocusResult,
)
from engine.backends.lookeng import (
    LooKengBackend,
    RunningContext,
    LemmaCacheEntry,
)

__all__ = [
    # Kimina
    "KiminaServerBackend",
    "KiminaServerClient",
    "BatchVerifyRequest",
    "BatchVerifyResult",
    # Pantograph
    "PantographBackend",
    "GoalFragment",
    "MVarFocusResult",
    # LooKeng
    "LooKengBackend",
    "RunningContext",
    "LemmaCacheEntry",
]

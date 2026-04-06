# Engine Legacy Modules

The following modules are **inactive legacy code** from APE v1.
They are retained for potential future use as planning models,
but are NOT called by any active code path.

**DO NOT add new dependencies on these modules.**

## Legacy modules (~2,400 lines)

- `engine/core/` — Expr, de Bruijn indices, Environment, LocalContext
- `engine/kernel/` — Heuristic TypeChecker (simplified Lean4 kernel in Python)
- `engine/tactic/` — 18 built-in tactics (Python reimplementation)
- `engine/state/` — ProofState, SearchTree, GoalState

## Active modules (APE v2)

- `engine/_core.py` — Shared pure functions (error classification, cache)
- `engine/lean_pool.py` — Lean4 REPL connection pool (sync)
- `engine/async_lean_pool.py` — Lean4 REPL connection pool (async + sync wrapper)
- `engine/broadcast.py` — Cross-direction broadcast bus
- `engine/prefilter.py` — L0 syntax pre-filter
- `engine/error_intelligence.py` — Structured AgentFeedback + exact?/apply?
- `engine/verification_scheduler.py` — L0→L1→L2 adaptive verification
- `engine/proof_session.py` — env_id state tree management
- `engine/incremental_verifier.py` — Incremental verification
- `engine/pool_scaler.py` — Dynamic pool scaling
- `engine/resource_scheduler.py` — Priority-based resource scheduling
- `engine/factory.py` — Component factory

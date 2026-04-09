"""common/ — Shared utilities used by both prover and agent layers.

This package contains pure data types, LLM prompt infrastructure, and
stateless utilities that are layer-agnostic. Both prover/ and agent/
depend on common/, establishing the dependency direction:

    engine ← common ← prover ← agent
                ↑                 ↑
                └─────────────────┘
"""

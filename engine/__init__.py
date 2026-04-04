"""APE — Agent Proof Engine

A proof verification engine designed from first principles for AI agent proof search.
Core innovations:
1. Persistent proof state (O(1) fork & backtrack via pyrsistent)
2. Layered verification (L0 quick / L1 elaborate / L2 certify)
3. Explicit constraint graph for parallel search
4. Agent-optimized state views (token-efficient)
5. Structured error feedback (machine-actionable)
"""
__version__ = "0.1.0"

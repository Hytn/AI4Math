"""
engine/lean_bridge — Lean4 Environment Bridge

Provides pre-built environments with core Lean4/Mathlib declarations.
This replaces a full .olean reader with a pragmatic approach:
declarations are defined in structured Python, matching Lean4's type system.

For production: replace with actual .olean parser or lean4export integration.
"""
from .prelude import build_prelude_env
from .minif2f_problems import MINIF2F_PROBLEMS

"""agent/brain/roles.py — Re-exports from common.roles for backward compatibility."""
from common.roles import AgentRole, ROLE_PROMPTS, MODEL_TIER_OVERRIDES, get_role_prompt

__all__ = ["AgentRole", "ROLE_PROMPTS", "MODEL_TIER_OVERRIDES", "get_role_prompt"]

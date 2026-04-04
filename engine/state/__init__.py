"""Persistent proof state — the core innovation for agent-first search.

Every ProofState is immutable. Tactics produce NEW states.
Fork is O(1). Backtrack is O(1). All states coexist in memory.
"""
from .proof_state import ProofState
from .meta_ctx import MetaContext
from .goal import Goal
from .search_tree import SearchTree, SearchNode, NodeId, NodeStatus
from .views import GoalView, TargetShape, format_goal_views_for_prompt

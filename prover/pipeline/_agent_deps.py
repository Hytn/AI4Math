"""prover/pipeline/_agent_deps.py — Agent layer dependency boundary

Centralizes ALL imports from agent/ that prover/pipeline/ needs.
Single integration seam between the prover and agent layers.

Pipeline modules import from here instead of directly from agent/.
This makes the boundary explicit, documented, and easy to refactor.

Dependency direction:
  engine < common < prover.core < prover.pipeline._agent_deps < agent
  engine < knowledge (knowledge may import engine, not vice versa)
"""

# -- Agent runtime --
from agent.runtime.sub_agent import (       # noqa: F401
    AgentSpec, AgentTask, AgentResult, ContextItem,
)
from agent.runtime.agent_pool import AgentPool           # noqa: F401
from agent.runtime.result_fuser import ResultFuser       # noqa: F401

# -- Agent strategy --
from agent.strategy.meta_controller import MetaController        # noqa: F401
from agent.strategy.confidence_estimator import ConfidenceEstimator  # noqa: F401
from agent.strategy.strategy_switcher import StrategySwitcher    # noqa: F401
from agent.strategy.direction_planner import (                   # noqa: F401
    ProofDirection, build_direction_prompt,
)

# -- Agent hooks & plugins --
from agent.hooks.hook_manager import HookManager         # noqa: F401
from agent.plugins.loader import PluginLoader            # noqa: F401

# -- Agent context --
from agent.context.error_summarizer import summarize_round_errors  # noqa: F401
from agent.context.context_window import ContextWindow   # noqa: F401

# -- Knowledge system --
from knowledge.store import UnifiedKnowledgeStore        # noqa: F401
from knowledge.writer import KnowledgeWriter             # noqa: F401
from knowledge.reader import KnowledgeReader             # noqa: F401
from knowledge.broadcaster import KnowledgeBroadcaster   # noqa: F401

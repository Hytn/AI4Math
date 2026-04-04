from dataclasses import dataclass
from engine.core import Expr, MetaId, LocalContext

@dataclass
class Goal:
    id: MetaId
    local_ctx: LocalContext
    target: Expr
    depth: int = 0

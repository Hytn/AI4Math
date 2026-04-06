# DEPRECATED: Legacy APE v1 module. Not called by any active code path.
# See engine/LEGACY.md for details. Do NOT add new dependencies.
from .expr import Expr, BinderInfo
from .name import Name
from .universe import Level
from .environment import Environment, ConstantInfo, InductiveInfo, ConstructorInfo, RecursorInfo
from .local_ctx import LocalContext, FVarId
from .meta import MetaId

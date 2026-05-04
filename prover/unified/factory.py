"""prover/unified/factory.py — Shared infrastructure-backend factory.

Both ``run_unified.py`` and ``run_eval.py`` need to construct the
optional Kimina / Pantograph / LooKeng backends in the same way before
handing them to ``UnifiedProofRunner``. This factory is the single
source of truth for that wiring.

Public API::

    profile_backend_hint(profile_name) -> Optional[str]
        Returns the implicit backend kind for profiles that need one
        (kimina_batch, pantograph_dsp, lookeng_lemma) or None.

    resolve_backend_kind(chosen, profile_name) -> str
        Honour --backend=auto by mapping profile → implicit backend.
        An explicit --backend overrides.

    build_infra_backends(kind, *, url=None, api_key=None) -> tuple
        Returns (kimina, pantograph, lookeng); any can be None.

    load_world_model(path) -> Optional[WorldModelPredictor]
    load_dialog_index(path) -> Optional[DialogIndex]
    load_knowledge(db_path) -> tuple[store, writer, reader]
        v11: optional-asset loaders extracted from the entrypoints. Each
        is fail-soft: returns None on failure with a warning, never raises.

History:
  v8 — each entrypoint inlined ~30 lines of copy-pasted backend wiring.
  v9 — unified backend construction here.
  v11 — also unified world_model / dialog_index / knowledge loading.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


_PROFILE_BACKEND_HINT = {
    "kimina_batch":   "kimina",
    "pantograph_dsp": "pantograph",
    "lookeng_lemma":  "lookeng",
}


def profile_backend_hint(profile_name: Optional[str]) -> Optional[str]:
    """Profile → implicit backend kind, or None if no implicit need."""
    if not profile_name:
        return None
    return _PROFILE_BACKEND_HINT.get(profile_name)


def resolve_backend_kind(chosen: str,
                         profile_name: Optional[str] = None) -> str:
    """Resolve the final backend kind.

    ``chosen == "auto"`` and a profile with an implicit hint → use the hint.
    Otherwise return ``chosen`` unchanged.
    """
    if chosen != "auto":
        return chosen
    hint = profile_backend_hint(profile_name)
    if hint is not None:
        logger.info(
            f"profile '{profile_name}' implies backend '{hint}', using it")
        return hint
    return chosen  # "auto"


async def build_infra_backends(
    kind: str,
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[object, object, object]:
    """Construct the optional backend trio (kimina, pantograph, lookeng).

    Returns ``(kimina, pantograph, lookeng)`` — any can be None. Each is
    only built when ``kind`` requests it, so a default ``--backend auto``
    run pays no cost for backends it doesn't use.

    The runner's tools tolerate any of these being None or in fallback —
    they register in fallback mode and return a structured "unavailable"
    error if the LLM tries to call them, rather than crashing the loop.
    """
    kimina = pantograph = lookeng = None

    if kind in ("kimina", "http"):
        from engine.backends.kimina_server import KiminaServerBackend
        kimina = KiminaServerBackend(base_url=url, api_key=api_key)
        await kimina.start()
        logger.info(
            f"Kimina backend started (fallback={kimina.is_fallback})")

    elif kind == "pantograph":
        from engine.backends.pantograph import PantographBackend
        pantograph = PantographBackend()
        await pantograph.start()
        logger.info(
            f"Pantograph backend started (mode={pantograph.mode})")

    elif kind == "lookeng":
        from engine.backends.lookeng import LooKengBackend
        lookeng = LooKengBackend()
        await lookeng.start()
        logger.info("LooKeng backend started")

    # 'local', 'socket', 'mock', 'fallback', 'auto' use the standard
    # lean_pool path and don't need any of the infrastructure backends.

    return kimina, pantograph, lookeng


# ═══════════════════════════════════════════════════════════════════════
# v11: Optional-asset loaders (previously inlined per entrypoint)
# ═══════════════════════════════════════════════════════════════════════


def load_world_model(path: Optional[str]):
    """Load an sklearn-backed WorldModelPredictor from .pkl, or None.

    Fail-soft: any error logs a warning and returns None. The runner
    treats world_model=None as "no gating".
    """
    if not path:
        return None
    try:
        from engine.world_model import make_world_model
        wm = make_world_model(path)
        logger.info(f"world model loaded from {path}")
        return wm
    except Exception as e:
        logger.warning(
            f"could not load world model from {path}: {e}. "
            f"Continuing without it.")
        return None


def load_dialog_index(path: Optional[str]):
    """Load a SQLite-backed DialogIndex, or None on failure."""
    if not path:
        return None
    try:
        from knowledge.dialog_index import DialogIndex
        idx = DialogIndex.load_from_sqlite(path)
        logger.info(
            f"dialog index loaded from {path} "
            f"({getattr(idx, 'size', '?')} entries)")
        return idx
    except Exception as e:
        logger.warning(
            f"could not load dialog index from {path}: {e}. "
            f"Continuing without it.")
        return None


def load_knowledge(db_path: Optional[str]):
    """Open a UnifiedKnowledgeStore + writer + reader at db_path.

    Returns ``(store, writer, reader)``. All three are None on failure.
    """
    if not db_path:
        return None, None, None
    try:
        from knowledge.store import UnifiedKnowledgeStore
        from knowledge.writer import KnowledgeWriter
        from knowledge.reader import KnowledgeReader
        store = UnifiedKnowledgeStore(db_path)
        writer = KnowledgeWriter(store)
        reader = KnowledgeReader(store)
        logger.info(f"knowledge store opened: {db_path}")
        return store, writer, reader
    except Exception as e:
        logger.warning(
            f"could not open knowledge store at {db_path}: {e}. "
            f"Continuing without it.")
        return None, None, None


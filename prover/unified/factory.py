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


# ═══════════════════════════════════════════════════════════════════════
# v15: factory loaders for the v14 features that previously had no
# CLI / no factory wiring. Without these, ``--policy-engine``,
# ``--plugins-dir`` and ``--lemma-bank-db`` could not be exposed
# through ``run_unified.py`` / ``run_eval.py`` and the v14 reservoirs
# stayed `policy_engine=None` / `plugin_loader=None` /
# `persistent_lemma_bank=None` in every default eval — i.e. v14 was
# unmeasurable in production.
# ═══════════════════════════════════════════════════════════════════════


def load_policy_engine(enabled: bool):
    """Build a ``PolicyEngine`` with the 5 default rules, or return None.

    Args:
        enabled: ``True`` to construct ``PolicyEngine.default()``;
                 anything else returns ``None`` (legacy v13 behaviour:
                 hard ``max_turns`` termination, no declarative rules).

    Why a boolean and not a path?
        Default ``PolicyRule`` set is pure-Python doctrine (no persisted
        state). A YAML / JSON config for selectively enabling rules is
        a future feature; for now the right granularity is "use defaults"
        vs "off". When you need fewer rules, hand-build an engine in code.

    Fail-soft: import error logs a warning and returns None — the
    runner continues with the v13 hardcoded path.
    """
    if not enabled:
        return None
    try:
        from engine.policy import PolicyEngine
        eng = PolicyEngine.default()
        logger.info(
            "policy engine: enabled (default rules: InfraRecovery, "
            "ConsecutiveSameError, BudgetEscalation, BankedLemmaDecompose, "
            "Reflection)")
        return eng
    except Exception as e:
        logger.warning(
            f"could not construct PolicyEngine: {e}. "
            f"Falling back to hardcoded max_turns termination.")
        return None


def load_plugin_loader(plugins_dir: Optional[str]):
    """Discover domain plugins from one or more directories.

    Args:
        plugins_dir: Path or comma-separated paths to plugin roots.
                     Each root contains ``<domain>/plugin.yaml`` etc.
                     Default layout: ``plugins/strategies/``.
                     None disables plugin injection entirely.

    Returns:
        ``PluginLoader`` with ``.discover()`` already called, or
        ``None`` if loading failed or the directory yielded no plugins.

    The runner's ``_build_initial_message`` calls
    ``loader.match(theorem)`` per problem; a None loader short-circuits
    to plain behaviour. So a None return is a legitimate eval state,
    not a bug.
    """
    if not plugins_dir:
        return None
    try:
        from prover.plugins import PluginLoader
        dirs = [d.strip() for d in plugins_dir.split(",") if d.strip()]
        if not dirs:
            return None
        loader = PluginLoader(plugin_dirs=dirs)
        loader.discover()
        n = len(getattr(loader, "_registry", {}))
        if n == 0:
            logger.warning(
                f"plugin loader: discovered 0 plugins in {dirs!r}. "
                f"Continuing without domain injection.")
            return None
        logger.info(f"plugin loader: {n} plugin(s) loaded from {dirs!r}")
        return loader
    except Exception as e:
        logger.warning(
            f"could not initialise PluginLoader from {plugins_dir!r}: "
            f"{e}. Continuing without domain injection.")
        return None


def load_persistent_lemma_bank(db_path: Optional[str],
                                 lean_version: Optional[str] = None,
                                 mathlib_rev: Optional[str] = None):
    """Open a SQLite-backed cross-problem lemma bank, or return None.

    Args:
        db_path:       Path to the SQLite file. None / empty disables.
        lean_version:  Optional Lean toolchain tag stamped on new lemmas
                       (e.g. ``"leanprover/lean4:v4.28.0"``). Used so
                       a future ``recheck_after_upgrade`` step can flag
                       lemmas extracted under a different toolchain.
        mathlib_rev:   Optional Mathlib commit hash, same purpose.

    Returns:
        ``PersistentLemmaBank`` or ``None``.

    The runner accepts ``persistent_lemma_bank=None`` as "no cross-problem
    sharing" — the in-memory ``LemmaBank`` per-problem is unaffected.
    """
    if not db_path:
        return None
    try:
        from prover.lemma_bank import PersistentLemmaBank
        bank = PersistentLemmaBank(
            db_path=db_path,
            lean_version=lean_version or "",
            mathlib_rev=mathlib_rev or "",
        )
        rev_short = (mathlib_rev or "?")[:12]
        logger.info(
            f"persistent lemma bank opened: {db_path} "
            f"(lean_version={lean_version or '?'}, mathlib_rev={rev_short})")
        return bank
    except Exception as e:
        logger.warning(
            f"could not open PersistentLemmaBank at {db_path!r}: {e}. "
            f"Continuing without cross-problem lemma reuse.")
        return None


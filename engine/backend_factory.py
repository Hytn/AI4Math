"""engine/backend_factory.py — Single entry point for choosing a Lean backend.

The project supports several Lean 4 verification backends now:

* ``local``       — bare ``LocalTransport`` running ``lean4-repl`` as a subprocess
* ``socket``      — ``SocketTransport`` connecting to ``docker/lean_daemon.py``
* ``http``        — ``HTTPTransport`` → Kimina Lean Server (REST API)
* ``kimina``      — alias for ``http``
* ``pantograph``  — ``PantographBackend`` (mvar focus, drafting, S-exp terms)
* ``lookeng``     — ``LooKengBackend`` (stateless + running-context lemma cache)
* ``mock``        — ``MockTransport`` (deterministic, for unit tests)

Pick one by name:

>>> from engine.backend_factory import build_backend
>>> backend = await build_backend("kimina", base_url="http://lean.local:8000")

The returned object always implements :class:`engine.transport.REPLTransport`,
so callers can drop it into ``AsyncLeanPool`` regardless of which backend
they chose.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from engine.transport import (
    REPLTransport, LocalTransport, SocketTransport,
    HTTPTransport, MockTransport, FallbackTransport,
)

logger = logging.getLogger(__name__)


SUPPORTED_BACKENDS = (
    "local", "socket", "http", "kimina", "pantograph", "lookeng",
    "mock", "fallback", "auto",
)


async def build_backend(
        kind: str = "auto",
        *,
        project_dir: str = ".",
        base_url: str = None,
        api_key: str = None,
        socket_path: str = None,
        timeout_seconds: int = 120,
        inner: Optional[REPLTransport] = None,
        inner_kind: Optional[str] = None) -> REPLTransport:
    """Construct a backend by name and start it.

    The function blocks until the backend's ``start()`` returns. If the
    requested backend isn't available (binary missing, server down,
    aiohttp not installed), the function does not raise — it returns
    a started ``REPLTransport`` that reports ``is_fallback=True``. This
    keeps benchmark scripts robust across environments.

    Pass ``kind="auto"`` to pick the first backend that successfully
    starts. The probe order is: ``local`` → ``socket`` → ``http`` →
    ``fallback``. ``pantograph`` and ``lookeng`` are *never* picked
    automatically because they imply a particular agent strategy; opt
    in explicitly.

    Backend chaining
    ----------------

    LooKeng is an outer/stateless wrapper that needs an inner verifier.
    When ``kind == "lookeng"``:

      * ``inner`` (a started ``REPLTransport``) is used directly if given.
      * Otherwise, ``inner_kind`` selects what to build for the inner —
        e.g. ``inner_kind="kimina"`` produces the production-grade
        chain (LooKeng's lemma cache → Kimina's REST batch → Lean REPL
        pool). The default ``inner_kind=None`` uses the in-process
        ``LocalTransport`` like before.
    """
    kind = (kind or "auto").lower()
    if kind not in SUPPORTED_BACKENDS:
        logger.warning(
            f"Unknown backend kind '{kind}'; falling back to 'auto'")
        kind = "auto"

    if kind == "auto":
        return await _build_auto(project_dir, timeout_seconds)

    if kind == "local":
        t = LocalTransport(project_dir=project_dir,
                            timeout_seconds=timeout_seconds)
        await t.start()
        return t

    if kind == "socket":
        path = socket_path or os.environ.get(
            "LEAN_SOCKET", "/workspace/exchange/lean.sock")
        t = SocketTransport(socket_path=path,
                              timeout_seconds=timeout_seconds)
        await t.start()
        return t

    if kind in ("http", "kimina"):
        t = HTTPTransport(base_url=base_url, api_key=api_key,
                            timeout_seconds=timeout_seconds * 5)
        await t.start()
        return t

    if kind == "pantograph":
        # Local import to keep this module's import cost low.
        from engine.backends.pantograph import PantographBackend
        t = PantographBackend(project_dir=project_dir,
                                timeout_seconds=timeout_seconds)
        await t.start()
        return t

    if kind == "lookeng":
        from engine.backends.lookeng import LooKengBackend
        # Build (or accept) the inner transport so LooKeng's outer
        # session/lemma-cache layer can wrap a real verifier.
        if inner is None and inner_kind:
            # Recursive build — but force the recursive call NOT to
            # re-enter the lookeng branch (would loop forever).
            if inner_kind == "lookeng":
                raise ValueError(
                    "lookeng cannot be its own inner backend")
            inner = await build_backend(
                kind=inner_kind,
                project_dir=project_dir,
                base_url=base_url,
                api_key=api_key,
                socket_path=socket_path,
                timeout_seconds=timeout_seconds)
        t = LooKengBackend(inner=inner, project_dir=project_dir)
        await t.start()
        return t

    if kind == "mock":
        t = MockTransport()
        await t.start()
        return t

    if kind == "fallback":
        t = FallbackTransport()
        await t.start()
        return t

    raise ValueError(f"Unhandled backend kind: {kind}")


async def _build_auto(project_dir: str,
                       timeout_seconds: int) -> REPLTransport:
    """Probe local/socket/http in order, fall back if all fail.

    A backend is considered "successful" if it starts and reports
    ``is_fallback=False``. We can't tell whether a real Lean is
    available without trying, so this probe sequence does exactly
    that — cheaply and in order of expected hit-rate.
    """
    # 1. LocalTransport — most common in dev.
    try:
        local = LocalTransport(project_dir=project_dir,
                                timeout_seconds=timeout_seconds)
        ok = await local.start()
        if ok and not local.is_fallback:
            logger.info("auto-backend: chose LocalTransport")
            return local
        await local.close()
    except Exception as e:
        logger.debug(f"auto-backend: local probe failed: {e}")

    # 2. SocketTransport — common in docker-compose deployments.
    sock_path = os.environ.get("LEAN_SOCKET")
    if sock_path and os.path.exists(sock_path):
        try:
            sock = SocketTransport(socket_path=sock_path,
                                    timeout_seconds=timeout_seconds)
            if await sock.start():
                logger.info("auto-backend: chose SocketTransport")
                return sock
        except Exception as e:
            logger.debug(f"auto-backend: socket probe failed: {e}")

    # 3. HTTPTransport — only when env var is set; we don't probe a
    #    default URL because that could surprise users.
    if os.environ.get("KIMINA_SERVER_URL"):
        try:
            http = HTTPTransport()
            if await http.start() and not http.is_fallback:
                logger.info("auto-backend: chose HTTPTransport (Kimina)")
                return http
            await http.close()
        except Exception as e:
            logger.debug(f"auto-backend: http probe failed: {e}")

    # 4. Fallback — fail-soft.
    logger.warning("auto-backend: no real backend available, using fallback")
    fb = FallbackTransport()
    await fb.start()
    return fb

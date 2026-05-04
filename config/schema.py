"""config/schema.py — 配置加载 + 校验 (v13 精简版)

v13: 之前 252 行, 校验 ~30 个字段, 但 80% 字段没有消费方。新版只校验
实际生效的字段, 砍掉所有装饰用的 schema 条目。
"""
from __future__ import annotations
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# 实际有消费方的字段 + 它们的合法范围。其他字段透传不校验。
_VALID_RANGES: dict[str, object] = {
    "agent.brain.provider": ["anthropic", "mock"],
    "prover.premise.mode": ["bm25", "embedding", "hybrid", "none"],
    "engine.lean_pool_size": (1, 64),
}


def load_config(path: str = "config/default.yaml",
                  overrides: dict = None) -> dict:
    """Load and validate configuration.

    Override priority (highest → lowest):
      1. ``overrides`` dict (programmatic)
      2. Environment variables (``AI4MATH_ENGINE__LEAN_POOL_SIZE=8``
         → ``engine.lean_pool_size=8``)
      3. Config file (YAML)
    """
    config: dict = {}
    p = Path(path)
    if p.exists():
        with open(p) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file not found: {path}, using defaults")

    _apply_env_overrides(config)

    if overrides:
        _deep_merge(config, overrides)

    _validate(config)

    # Populate flat alias for callers used to reading top-level keys.
    config["lean_pool_size"] = (
        config.get("engine", {}).get("lean_pool_size", 4))

    return config


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _apply_env_overrides(config: dict) -> None:
    """``AI4MATH_FOO__BAR=1`` → ``config['foo']['bar'] = 1`` (with type coerce).

    Backward-compat shim: ``APE_FOO__BAR`` accepted with deprecation log.
    """
    for env_key, env_val in os.environ.items():
        canonical_key = None
        if env_key.startswith("AI4MATH_"):
            canonical_key = env_key[len("AI4MATH_"):]
        elif env_key.startswith("APE_"):
            canonical_key = env_key[len("APE_"):]
            logger.info(
                f"env var {env_key} uses deprecated APE_* prefix; "
                f"use AI4MATH_* instead")
        if not canonical_key:
            continue
        path = canonical_key.lower().split("__")
        cur = config
        for seg in path[:-1]:
            cur = cur.setdefault(seg, {})
        cur[path[-1]] = _coerce(env_val)


def _coerce(s: str):
    """Best-effort string → int/float/bool/str."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _validate(config: dict) -> None:
    """Validate every key in ``_VALID_RANGES`` against config; warn on bad value."""
    for dotted, spec in _VALID_RANGES.items():
        v = _get_dotted(config, dotted)
        if v is None:
            continue
        if isinstance(spec, list):
            if v not in spec:
                logger.warning(
                    f"config: {dotted}={v!r} not in allowed values {spec}")
        elif isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            if not (isinstance(v, (int, float)) and lo <= v <= hi):
                logger.warning(
                    f"config: {dotted}={v!r} not in range [{lo}, {hi}]")


def _get_dotted(d: dict, dotted: str):
    cur = d
    for seg in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(seg)
        if cur is None:
            return None
    return cur

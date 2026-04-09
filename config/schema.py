"""config/schema.py — 配置 schema 校验"""
from __future__ import annotations
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Required fields and their types
_REQUIRED_SECTIONS = {
    "agent": {
        "brain": {"provider": str, "model": str},
    },
    "prover": {
        "pipeline": {"max_samples": int},
    },
    "engine": {
        # engine section is required but individual fields have defaults
    },
}

# Valid value ranges — ALL keys use dotted nested paths
_VALID_RANGES = {
    "agent.brain.provider": ["anthropic", "mock"],
    "prover.pipeline.max_samples": (1, 10000),
    "prover.pipeline.samples_per_round": (1, 128),
    "prover.pipeline.max_workers": (1, 32),
    "prover.pipeline.temperature": (0.0, 2.0),
    "prover.premise.mode": ["bm25", "embedding", "hybrid", "none"],
    "prover.verifier.timeout_seconds": (1, 3600),
    "prover.verifier.mode": ["docker", "local", "mock"],
    # APE v2 engine parameters
    "engine.lean_pool_size": (1, 64),
    "engine.max_wall_seconds": (1, 86400),
    "engine.pipeline_queue_size": (1, 256),
    "engine.num_verifiers": (1, 64),
    # Pool scaler
    "engine.pool_scaler.min_sessions": (1, 64),
    "engine.pool_scaler.max_sessions": (1, 128),
    "engine.pool_scaler.scale_up_threshold": (0.0, 1.0),
    "engine.pool_scaler.scale_down_threshold": (0.0, 1.0),
    "engine.pool_scaler.cooldown_seconds": (0.0, 600.0),
    # Agent strategy
    "agent.strategy.reflection_interval": (1, 100),
}

# Backward-compatible aliases: flat key → canonical dotted path
# Allows config.get("lean_pool_size") to resolve from engine.lean_pool_size
_KEY_ALIASES = {
    "lean_pool_size": "engine.lean_pool_size",
    "lean_project_dir": "prover.verifier.lean_project_dir",
    "max_workers": "prover.pipeline.max_workers",
    "max_samples": "prover.pipeline.max_samples",
    "max_wall_seconds": "engine.max_wall_seconds",
    "reflection_interval": "agent.strategy.reflection_interval",
    "pipeline_queue_size": "engine.pipeline_queue_size",
    "num_verifiers": "engine.num_verifiers",
}


def load_config(path: str = "config/default.yaml",
                overrides: dict = None) -> dict:
    """Load and validate configuration.

    Override priority (highest → lowest):
      1. overrides dict (programmatic)
      2. Environment variables (APE_ENGINE__LEAN_POOL_SIZE=8 → engine.lean_pool_size=8)
      3. Config file (YAML)
    """
    config = {}
    p = Path(path)
    if p.exists():
        with open(p) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file not found: {path}, using defaults")

    # Apply environment variable overrides
    _apply_env_overrides(config)

    if overrides:
        _deep_merge(config, overrides)

    # Populate flat aliases from nested values (backward compat)
    _populate_aliases(config)

    # Validate
    issues = validate_config(config)
    critical = [i for i in issues if i.startswith("Missing required")]
    warnings = [i for i in issues if i not in critical]

    for issue in warnings:
        logger.warning(f"Config issue: {issue}")

    if critical:
        msg = "Configuration has critical issues:\n" + "\n".join(f"  - {c}" for c in critical)
        raise ValueError(msg)

    return config


def validate_config(config: dict) -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Check required sections exist
    for section, fields in _REQUIRED_SECTIONS.items():
        if section not in config:
            issues.append(f"Missing required section: {section}")
            continue
        for subsection, required_fields in fields.items():
            if subsection not in config[section]:
                issues.append(
                    f"Missing required subsection: {section}.{subsection}")
                continue
            for field_name, field_type in required_fields.items():
                val = config[section][subsection].get(field_name)
                if val is not None and not isinstance(val, field_type):
                    issues.append(
                        f"{section}.{subsection}.{field_name}: "
                        f"expected {field_type.__name__}, got {type(val).__name__}")

    # Check value ranges (nested keys)
    for dotted_key, valid in _VALID_RANGES.items():
        val = _get_nested(config, dotted_key)
        if val is None:
            continue
        if isinstance(valid, list):
            if val not in valid:
                issues.append(
                    f"{dotted_key}: '{val}' not in valid values {valid}")
        elif isinstance(valid, tuple) and len(valid) == 2:
            lo, hi = valid
            if isinstance(val, (int, float)) and not (lo <= val <= hi):
                issues.append(
                    f"{dotted_key}: {val} outside valid range [{lo}, {hi}]")

    # Also validate flat keys via alias resolution (backward compat)
    for flat_key, dotted_path in _KEY_ALIASES.items():
        val = config.get(flat_key)
        if val is None:
            continue
        valid = _VALID_RANGES.get(dotted_path)
        if valid is None:
            continue
        if isinstance(valid, list):
            if val not in valid:
                issues.append(
                    f"{flat_key}: '{val}' not in valid values {valid}")
        elif isinstance(valid, tuple) and len(valid) == 2:
            lo, hi = valid
            if isinstance(val, (int, float)) and not (lo <= val <= hi):
                issues.append(
                    f"{flat_key}: {val} outside valid range [{lo}, {hi}]")

    return issues


def _populate_aliases(config: dict):
    """Copy nested values to flat top-level keys for backward compatibility.

    If config has engine.lean_pool_size=8, also sets config["lean_pool_size"]=8
    so that config.get("lean_pool_size", default) works in factory code.
    Flat keys already present in config take precedence (explicit override).
    """
    for flat_key, dotted_path in _KEY_ALIASES.items():
        if flat_key not in config:
            val = _get_nested(config, dotted_path)
            if val is not None:
                config[flat_key] = val


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _get_nested(d: dict, dotted_key: str):
    """Get a value from a nested dict using dotted key path."""
    keys = dotted_key.split(".")
    current = d
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return None
        current = current[k]
    return current


def _set_nested(d: dict, dotted_key: str, value):
    """Set a value in a nested dict using dotted key path."""
    keys = dotted_key.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def _apply_env_overrides(config: dict, prefix: str = "APE_"):
    """Apply environment variable overrides.

    Convention: APE_ENGINE__LEAN_POOL_SIZE=8 → engine.lean_pool_size=8
      - Prefix: APE_ (configurable)
      - Double underscore (__) separates nesting levels
      - Single underscore (_) within a level name

    Type coercion: attempts int → float → bool → string.
    """
    import os
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Strip prefix, convert to dotted path
        raw = key[len(prefix):]
        # Double underscore → nesting separator
        parts = raw.split("__")
        dotted = ".".join(p.lower() for p in parts)

        # Type coercion
        coerced = _coerce_value(value)

        _set_nested(config, dotted, coerced)
        logger.info(f"Config override from env: {dotted}={coerced}")


def _coerce_value(value: str):
    """Coerce string env var to appropriate Python type.

    Priority: None → int → float → bool → string.
    Integer parsing comes before boolean so that "0" → 0 and "1" → 1,
    not False/True.
    """
    # None
    if value.lower() in ("null", "none", ""):
        return None
    # Integer (must come before boolean so "0"→0, "1"→1)
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    # Boolean (only non-numeric words)
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    # String
    return value

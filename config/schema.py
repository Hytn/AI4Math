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
}

# Valid value ranges
_VALID_RANGES = {
    "agent.brain.provider": ["anthropic", "mock"],
    "prover.pipeline.max_samples": (1, 10000),
    "prover.pipeline.samples_per_round": (1, 128),
    "prover.pipeline.max_workers": (1, 32),
    "prover.pipeline.temperature": (0.0, 2.0),
    "prover.premise.mode": ["bm25", "embedding", "hybrid", "none"],
    "prover.verifier.timeout_seconds": (1, 3600),
}


def load_config(path: str = "config/default.yaml",
                overrides: dict = None) -> dict:
    """Load and validate configuration."""
    config = {}
    p = Path(path)
    if p.exists():
        with open(p) as f:
            config = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file not found: {path}, using defaults")

    if overrides:
        _deep_merge(config, overrides)

    # Validate
    issues = validate_config(config)
    for issue in issues:
        logger.warning(f"Config issue: {issue}")

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

    # Check value ranges
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

    return issues


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

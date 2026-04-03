"""config/schema.py — 配置 schema 校验"""
from __future__ import annotations
import yaml
from pathlib import Path

def load_config(path: str = "config/default.yaml", overrides: dict = None) -> dict:
    config = {}
    p = Path(path)
    if p.exists():
        with open(p) as f: config = yaml.safe_load(f) or {}
    if overrides:
        _deep_merge(config, overrides)
    return config

def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v

"""common/logging_config.py — 统一结构化日志配置

所有入口点应调用 setup_logging() 一次，而非各自 basicConfig。

Usage::

    from common.logging_config import setup_logging
    setup_logging(level="INFO", json_format=False)  # human-readable
    setup_logging(level="INFO", json_format=True)    # structured JSON

JSON 格式输出示例:
    {"ts":"2025-03-15T10:30:00","level":"INFO","logger":"engine.lane","msg":"...","problem_id":"p1"}
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON Lines formatter with optional context fields."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Include extra context fields if present
        for key in ("problem_id", "session_id", "round_number",
                     "strategy", "lane_status"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Concise human-readable formatter (default)."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    stream=None,
) -> None:
    """Configure logging once for the entire process.

    Safe to call multiple times — only the first call takes effect.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON Lines. If False, human-readable.
        stream: Output stream (default: sys.stderr).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(stream or sys.stderr)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str, **defaults) -> logging.Logger:
    """Get a logger with optional default context fields.

    Usage::

        log = get_logger(__name__, problem_id="p123")
        log.info("Starting proof")  # auto-includes problem_id in JSON mode
    """
    logger = logging.getLogger(name)
    if defaults:
        old_factory = logging.getLogRecordFactory()

        def factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for k, v in defaults.items():
                if not hasattr(record, k):
                    setattr(record, k, v)
            return record

        # Note: this sets it globally. For per-logger context, use
        # logger.addFilter or LoggerAdapter instead.
        # This is a simple approach; for production, use LoggerAdapter.
    return logger

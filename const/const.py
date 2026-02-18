"""Centralized constants used by clawflow.

Many of these values can be overridden by environment variables.
"""

from __future__ import annotations

import os


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return default if v is None else v


# LLM / agent knobs
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_ITERATIONS = _env_int("MAX_ITERATIONS", 20)
MAX_TOOL_CONSECUTIVE_COUNT = _env_int("MAX_TOOL_CONSECUTIVE_COUNT", 5)


# Logging / skill loading
TRUNCATE_LOG_LEN = _env_int("TRUNCATE_LOG_LEN", 4000)


import json
import logging
import os
from typing import Dict, Any

# Configuration management
CONFIG = None
CONFIG_PATHS = [
    "config.yaml",
    ".env",
    os.path.expanduser("./config.yaml"),
]

logger = logging.getLogger(__name__)


def _load_config() -> Dict[str, Any]:
    """Load configuration from file(s) and environment variables."""
    global CONFIG
    if CONFIG is not None:
        return CONFIG

    config = {}

    # 1. Load from config files (first found)
    for path in CONFIG_PATHS:
        if os.path.isfile(path):
            try:
                if path.endswith((".yaml")):
                    try:
                        import yaml

                        with open(path, "r", encoding="utf-8") as f:
                            file_config = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError(
                            "PyYAML not installed. Install with `pip install PyYAML`"
                        )
                elif path == ".env":
                    # Simple .env file support (key=value)
                    file_config = {}
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                if "=" in line:
                                    key, val = line.split("=", 1)
                                    file_config[key.strip()] = val.strip()
                else:
                    continue

                # Merge (later files override earlier ones)
                if isinstance(file_config, dict):
                    _deep_update(config, file_config)
                break  # Use first found config file
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to load config file {path}: {e}"
                )
                continue

    # 2. Environment variables override config file
    # Convert environment variables with prefix AGENT_ to nested dict
    env_config = {}
    for key, val in os.environ.items():
        if key.startswith("AGENT_"):
            # Convert AGENT_LLM_MODEL to {"llm": {"model": val}}
            parts = key.lower().split("_")
            current = env_config
            for part in parts[1:-1]:  # skip AGENT and last part
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = val
    _deep_update(config, env_config)

    CONFIG = config
    return config


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively merge source into target."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value by dot-separated key (e.g., 'llm.model')."""
    config = _load_config()
    if key is None:
        return config

    parts = key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def env(name: str, default: str | None = None) -> str | None:
    # 1. Check environment variable (original behavior)
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return v

    # 2. Try to get from config file with dot notation
    key = name.lower()
    # Keep original name as key (e.g., "OPENAI_API_KEY" -> "openai.api.key")
    # key = key.replace("_", ".")

    config_value = get_config(key)
    if config_value is not None:
        return str(config_value)

    # 3. Fallback to default
    return default

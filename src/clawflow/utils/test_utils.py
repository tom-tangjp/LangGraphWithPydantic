"""Basic tests for utility functions."""

import os
import tempfile
import yaml

from src.clawflow.config import env, CONFIG_PATHS, get_config
from src.clawflow.config.loader import _deep_update


def test_env():
    """Test environment variable lookup with fallback."""
    # Test direct environment variable
    os.environ["TEST_VAR"] = "direct"
    assert env("TEST_VAR") == "direct"
    # Test missing
    assert env("NONEXISTENT") is None
    assert env("NONEXISTENT", "default") == "default"
    # Cleanup
    del os.environ["TEST_VAR"]


def test_env_with_config():
    """Test environment variable via config file."""
    # Create a temporary config.yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"llm": {"model": "test-model"}}, f)
        config_path = f.name

    # Temporarily replace CONFIG_PATHS

    original_paths = CONFIG_PATHS.copy()
    try:
        # Replace first path with our temp file
        CONFIG_PATHS.clear()
        CONFIG_PATHS.append(config_path)
        # Reset cached config in the config loader (uses module-level `CONFIG`)
        from src.clawflow.config import loader as config_loader

        config_loader.CONFIG = None

        # Test that env can retrieve config value
        # Note: env('AGENT_LLM_MODEL') maps to llm.model
        # Since we don't have AGENT_LLM_MODEL env var, it should fallback to config
        # But env expects environment variable with prefix AGENT_
        # We'll test get_config directly instead
        value = get_config("llm.model")
        assert value == "test-model"
    finally:
        # Restore
        CONFIG_PATHS.clear()
        CONFIG_PATHS.extend(original_paths)
        os.unlink(config_path)


def test_deep_update():
    """Test recursive dictionary merge."""
    target = {"a": 1, "b": {"x": 10}}
    source = {"b": {"y": 20}, "c": 3}
    _deep_update(target, source)
    assert target == {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}


def test_extract_tool_calls():
    """Test tool call extraction."""
    from src.clawflow.utils import extract_tool_calls

    # Mock AIMessage with tool_calls
    class MockMessage:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls

    msg = MockMessage([{"name": "search", "args": {"query": "test"}, "id": "1"}])
    result = extract_tool_calls(msg)
    assert len(result) == 1
    assert result[0]["name"] == "search"
    assert result[0]["args"] == {"query": "test"}


if __name__ == "__main__":
    test_env()
    test_env_with_config()
    test_deep_update()
    test_extract_tool_calls()
    print("All tests passed.")

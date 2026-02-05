"""
mcp_shell
Dangerous shell execution (run_bash). Disabled by default.

Transport: stdio (for LangGraph Host spawning this process).
Safety: logs go to stderr; stdout is reserved for MCP protocol.
"""
import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_shell")

mcp = FastMCP("mcp_shell")

# Your existing LangChain tools module (must be importable in this environment)
import tools as _tools

def _invoke(tool_name: str, **kwargs) -> str:
    """Invoke a LangChain tool (from tools.py) via TOOL_REGISTRY."""
    try:
        tool = _tools.TOOL_REGISTRY[tool_name]
    except Exception as e:
        return json.dumps({"ok": False, "error": f"tool not found: {tool_name} ({e})"}, ensure_ascii=False)
    try:
        return tool.invoke(kwargs)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"invoke {tool_name} failed: {e}"}, ensure_ascii=False)

_SHELL_ENABLED = os.environ.get("MCP_SHELL_ENABLED", "0") == "1"
_PREFIXES = [p.strip() for p in os.environ.get("MCP_SHELL_ALLOW_PREFIXES", "").split(",") if p.strip()]

def _cap_timeout(timeout: int, max_s: int = 300) -> int:
    try:
        t = int(timeout)
    except Exception:
        t = 60
    return max(1, min(t, max_s))

def _prefix_allowed(cmd: str) -> bool:
    if not _PREFIXES:
        return True
    s = (cmd or "").lstrip()
    return any(s.startswith(p) for p in _PREFIXES)
@mcp.tool()
def run_bash(
    command: str, timeout: int = 60, work_dir: str = "."
) -> str:
    """Run bash (requires MCP_SHELL_ENABLED=1)."""
    if not _SHELL_ENABLED:
        return json.dumps({"ok": False, "error": "shell disabled (set MCP_SHELL_ENABLED=1)"}, ensure_ascii=False)
    if not _prefix_allowed(command):
        return json.dumps({"ok": False, "error": "command blocked by MCP_SHELL_ALLOW_PREFIXES"}, ensure_ascii=False)
    timeout2 = _cap_timeout(timeout, 300)
    return _invoke("run_bash", command=command, timeout=timeout2, work_dir=work_dir)

if __name__ == "__main__":
    mcp.run(transport="stdio")

"""
mcp_exec_safe
Controlled execution tools. No arbitrary shell. Optional mutation gates for formatters.

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
logger = logging.getLogger("mcp_exec_safe")

mcp = FastMCP("mcp_exec_safe")

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

_ALLOW_MUTATION = os.environ.get("MCP_ALLOW_SOURCE_MUTATION", "0") == "1"
_ALLOW_PIP = os.environ.get("MCP_ALLOW_PIP", "0") == "1"
_ALLOW_PYTHON_MODULES = os.environ.get("MCP_ALLOW_PYTHON_MODULES", "0") == "1"

# Additional restrictions layered on top of tools.py SAFE_COMMANDS
_DEFAULT_ALLOWED_BASE = {"pytest", "flake8", "black", "isort", "python"}  # 'pip' disabled by default

def _parse_cmd(command: str) -> List[str]:
    import shlex
    try:
        return shlex.split(command)
    except Exception:
        return []

def _is_allowed_run_safe_command(command: str) -> Union[bool, str]:
    parts = _parse_cmd(command)
    if not parts:
        return "empty command"
    base = parts[0]

    if base == "pip" and not _ALLOW_PIP:
        return "pip is disabled (set MCP_ALLOW_PIP=1 to enable)"
    if base not in _DEFAULT_ALLOWED_BASE and base != "pip":
        return f"base command not allowed: {base}"

    if base == "python" and not _ALLOW_PYTHON_MODULES:
        # Only allow: python -m py_compile <files...>
        if len(parts) >= 3 and parts[1] == "-m" and parts[2] == "py_compile":
            return True
        return "python modules are restricted (set MCP_ALLOW_PYTHON_MODULES=1 to allow other -m modules)"

    return True

def _cap_timeout(timeout: int, max_s: int = 120) -> int:
    try:
        t = int(timeout)
    except Exception:
        t = 30
    return max(1, min(t, max_s))
@mcp.tool()
def run_safe_command(
    command: str, timeout: int = 30
) -> str:
    """Run a whitelisted command (extra gated)."""
    ok = _is_allowed_run_safe_command(command)
    if ok is not True:
        return json.dumps({"ok": False, "error": str(ok), "output": ""}, ensure_ascii=False)
    timeout2 = _cap_timeout(timeout, 120)
    return _invoke("run_safe_command", command=command, timeout=timeout2)

@mcp.tool()
def analyze_code(
    path: str = ".", tool_name: str = "flake8"
) -> str:
    """Controlled execution proxy."""
    return _invoke("analyze_code", **locals())

@mcp.tool()
def run_tests(
    path: str = "tests", options: str = "-v"
) -> str:
    """Controlled execution proxy."""
    return _invoke("run_tests", **locals())

@mcp.tool()
def check_syntax(
    path: str
) -> str:
    """Controlled execution proxy."""
    return _invoke("check_syntax", **locals())

@mcp.tool()
def run_cpp_linter(
    path: str, tool_name: str = "clang-tidy"
) -> str:
    """Controlled execution proxy."""
    return _invoke("run_cpp_linter", **locals())

@mcp.tool()
def check_cpp_syntax(
    path: str, compiler: str = "clang++", std: str = "c++17"
) -> str:
    """Controlled execution proxy."""
    return _invoke("check_cpp_syntax", **locals())

@mcp.tool()
def format_code(
    path: str = ".", formatter: str = "black"
) -> str:
    """Format code (gated)."""
    if not _ALLOW_MUTATION:
        return json.dumps({"ok": False, "error": "format_code disabled (set MCP_ALLOW_SOURCE_MUTATION=1)"}, ensure_ascii=False)
    return _invoke("format_code", **locals())


@mcp.tool()
def format_cpp(
    path: str, style: str = "llvm"
) -> str:
    """Format C++ (gated)."""
    if not _ALLOW_MUTATION:
        return json.dumps({"ok": False, "error": "format_cpp disabled (set MCP_ALLOW_SOURCE_MUTATION=1)"}, ensure_ascii=False)
    return _invoke("format_cpp", **locals())



if __name__ == "__main__":
    mcp.run(transport="stdio")

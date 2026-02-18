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
from typing import List, Union

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_exec_safe")

mcp = FastMCP("mcp_exec_safe")

# Your existing LangChain tools module (must be importable in this environment)
try:
    # When installed as a package (preferred).
    from my_tools import tools as _tools
except Exception:  # pragma: no cover
    # When executed from this folder as a script.
    # Avoid ambiguous `import tools` (can resolve to this repo's top-level `tools/` dir or another package).
    # Ensure `tools/src` is on sys.path, then import the canonical module path.
    from pathlib import Path
    _src_root = str(Path(__file__).resolve().parents[1])  # .../tools/src
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    from my_tools import tools as _tools


def _invoke(tool_name: str, **kwargs) -> str:
    """Invoke a LangChain tool (from tools.py) via TOOL_REGISTRY."""
    try:
        tool = _tools.TOOL_REGISTRY[tool_name]
    except Exception as e:
        return json.dumps({"ok": False, "error": f"tool not found: {tool_name} ({e})"}, ensure_ascii=False)
    try:
        res = tool.invoke(kwargs)
        # FastMCP tool schema here is declared as `-> str` (see @mcp.tool() functions below).
        # Some LangChain tools may return structured Python objects (list/dict). Normalize them
        # into a JSON string to avoid downstream schema validation errors.
        if isinstance(res, str):
            return res
        if isinstance(res, (bytes, bytearray)):
            try:
                return res.decode("utf-8", errors="replace")
            except Exception:
                return str(res)
        try:
            return json.dumps(res, ensure_ascii=False)
        except TypeError:
            return str(res)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"invoke {tool_name} failed: {e}"}, ensure_ascii=False)

_ALLOW_MUTATION = os.environ.get("MCP_ALLOW_SOURCE_MUTATION", "0") == "1"
_ALLOW_PIP = os.environ.get("MCP_ALLOW_PIP", "0") == "1"
_ALLOW_PIP_READONLY = os.environ.get("MCP_ALLOW_PIP_READONLY", "0") == "1"
_ALLOW_PIP_WRITE = os.environ.get("MCP_ALLOW_PIP_WRITE", "0") == "1"
_ALLOW_PYTHON_MODULES = os.environ.get("MCP_ALLOW_PYTHON_MODULES", "0") == "1"
_ALLOW_PYTHON_C = os.environ.get("MCP_ALLOW_PYTHON_C", "0") == "1"
_ALLOW_GO = os.environ.get("MCP_ALLOW_GO", "0") == "1"
_ALLOW_ALL_COMMANDS = os.environ.get("MCP_ALLOW_ALL_COMMANDS", "0") == "1"

# Additional restrictions layered on top of tools.py SAFE_COMMANDS
_DEFAULT_ALLOWED_BASE = {"pytest", "flake8", "black", "isort", "python", "python3", "go"}  # 'pip' disabled by default

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

    if _ALLOW_ALL_COMMANDS:
        return True

    base = parts[0]

    # pip: default disabled; can enable read-only or write explicitly
    if base == "pip":
        if len(parts) == 2 and parts[1] in ("--version", "-V"):
            return True if _ALLOW_PIP_READONLY or _ALLOW_PIP_WRITE or _ALLOW_PIP else "pip is disabled (set MCP_ALLOW_PIP_READONLY=1)"
        if len(parts) < 2:
            return "pip missing subcommand"
        sub = parts[1]
        if sub in ("show", "list", "freeze"):
            return True if _ALLOW_PIP_READONLY or _ALLOW_PIP_WRITE or _ALLOW_PIP else "pip is disabled (set MCP_ALLOW_PIP_READONLY=1)"
        if sub in ("install", "uninstall"):
            return True if (_ALLOW_PIP_WRITE or _ALLOW_PIP) else "pip write ops disabled (set MCP_ALLOW_PIP_WRITE=1)"
        return f"pip subcommand not allowed: {sub}"


    if base == "pip" and not _ALLOW_PIP:
        return "pip is disabled (set MCP_ALLOW_PIP=1 to enable)"

    if base not in _DEFAULT_ALLOWED_BASE and base != "pip":
        return f"base command not allowed: {base}"

    if base in ("python", "python3") and not _ALLOW_PYTHON_MODULES:
        # Allow: python --version / -V
        if len(parts) == 2 and parts[1] in ("--version", "-V"):
            return True
        # Optional: python -c ... (allowlisted)
        if len(parts) >= 3 and parts[1] == "-c":
            return True if _ALLOW_PYTHON_C else "python -c disabled (set MCP_ALLOW_PYTHON_C=1)"
            # Only allow: python -m py_compile <files...> OR python -m pytest ...
        if len(parts) >= 3 and parts[1] == "-m" and parts[2] in ("py_compile", "pytest"):
            return True
        return "python modules are restricted (set MCP_ALLOW_PYTHON_MODULES=1 to allow other -m modules)"

    if base == "go" and  not _ALLOW_GO:
        return "go is disabled (set MCP_ALLOW_GO=1 to enable)"

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
    """Run a guarded shell command from a small allowlist.

    Allowed by default:
    - python / python3, pytest
    - flake8, black, isort

    Optional (disabled by default):
    - go subcommands (version/env/list/test/vet) if MCP_ALLOW_GO=1
      - Note: go env -w is blocked.

    Args:
    - command: full command line string
    - timeout: seconds (capped)

    Returns:
    - JSON string: {"ok": bool, "error": str, "output": str, "command": str}
    """
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

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

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_shell")

mcp = FastMCP("mcp_shell")

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

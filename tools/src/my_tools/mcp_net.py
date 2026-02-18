"""
mcp_net
Network tools (web_search/web_open/http_get). Disabled by default with domain allowlist.

Transport: stdio (for LangGraph Host spawning this process).
Safety: logs go to stderr; stdout is reserved for MCP protocol.
"""
import os
import sys
import json
import logging
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logger = logging.getLogger("mcp_net")

mcp = FastMCP("mcp_net")

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
        # MCP tool schema here is declared as `-> str` (see @mcp.tool() functions below).
        # Some LangChain tools return structured Python objects (list/dict). Normalize them
        # into a JSON string so downstream schema validation won't fail, while keeping
        # multi-result structure intact (callers can `json.loads` if needed).
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
            # As a last resort, fall back to string representation.
            return str(res)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"invoke {tool_name} failed: {e}"}, ensure_ascii=False)

_NET_ENABLED = os.environ.get("MCP_NET_ENABLED", "0") == "1"
_ALLOW_ALL = os.environ.get("MCP_NET_ALLOW_ALL", "0") == "1"
_ALLOW_DOMAINS = [d.strip().lower() for d in os.environ.get("MCP_NET_ALLOW_DOMAINS", "").split(",") if d.strip()]

def _deny_if_disabled() -> Optional[str]:
    if not _NET_ENABLED:
        return json.dumps({"ok": False, "error": "network disabled (set MCP_NET_ENABLED=1)"}, ensure_ascii=False)
    return None

def _domain_allowed(url: str) -> Union[bool, str]:
    if _ALLOW_ALL:
        return True
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        if not host:
            return "missing hostname"
    except Exception:
        return "invalid url"
    if not _ALLOW_DOMAINS:
        return "no allowlist (set MCP_NET_ALLOW_DOMAINS=example.com or MCP_NET_ALLOW_ALL=1)"
    for d in _ALLOW_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return f"domain not allowed: {host}"


@mcp.tool()
def web_search(
    query: str, max_results: int = 5
) -> str:
    """Web search (gated)."""
    deny = _deny_if_disabled()
    if deny:
        return deny
    # best-effort argument mapping
    try:
        return _invoke("web_search", query=query, max_results=max_results)
    except Exception:
        return _invoke("web_search", q=query, max_results=max_results)


@mcp.tool()
def web_open(
    url: str, max_chars: int = 20000
) -> str:
    """Open URL (gated + allowlist)."""
    deny = _deny_if_disabled()
    if deny:
        return deny
    ok = _domain_allowed(url)
    if ok is not True:
        return json.dumps({"ok": False, "error": str(ok)}, ensure_ascii=False)
    return _invoke("web_open", url=url, max_chars=max_chars)


@mcp.tool()
def http_get(
    url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30
) -> str:
    """HTTP GET (gated + allowlist)."""
    deny = _deny_if_disabled()
    if deny:
        return deny
    ok = _domain_allowed(url)
    if ok is not True:
        return json.dumps({"ok": False, "error": str(ok)}, ensure_ascii=False)
    return _invoke("http_get", url=url, headers=headers, timeout=timeout)



if __name__ == "__main__":
    mcp.run(transport="stdio")

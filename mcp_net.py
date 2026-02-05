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
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_net")

mcp = FastMCP("mcp_net")

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

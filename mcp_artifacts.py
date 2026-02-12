"""
mcp_artifacts
Write artifacts only. All writes are forced under artifacts/ (or MCP_ARTIFACTS_PREFIX).

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

import utils

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_artifacts")

mcp = FastMCP("mcp_artifacts")

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

_ARTIFACTS_PREFIX = os.environ.get("MCP_ARTIFACTS_PREFIX", "artifacts").strip().strip("/")

def _normalize_rel(p: str) -> str:
    p = (p or "").replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p

def _coerce_artifacts_path(p: str) -> str:
    p = _normalize_rel(p)
    if not p:
        return _ARTIFACTS_PREFIX
    if p.startswith("/") or p.startswith("~"):
        raise ValueError("absolute paths are not allowed")
    # block parent traversal
    if p.startswith("..") or "/../" in f"/{p}/":
        raise ValueError("path traversal is not allowed")
    # already under prefix?
    if p == _ARTIFACTS_PREFIX or p.startswith(_ARTIFACTS_PREFIX + "/"):
        return p
    return f"{_ARTIFACTS_PREFIX}/{p}"

@utils.timer
@mcp.tool()
def ensure_dir(
    path: str
) -> str:
    """Create directories under artifacts/."""
    try:
        safe_path = _coerce_artifacts_path(path)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return _invoke("ensure_dir", path=safe_path)

@utils.timer
@mcp.tool()
def write_text_file(
    path: str, content: str, mode: str = "overwrite", encoding: str = "utf-8"
) -> str:
    """Write text files under artifacts/."""
    try:
        safe_path = _coerce_artifacts_path(path)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return _invoke("write_text_file", path=safe_path, content=content, mode=mode, encoding=encoding)

@utils.timer
@mcp.tool()
def save_mermaid_diagram(
    mermaid_code: str, filename: str = "diagram.md"
) -> str:
    """Save Mermaid diagram under artifacts/."""
    try:
        safe_filename = _coerce_artifacts_path(filename)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return _invoke("save_mermaid_diagram", mermaid_code=mermaid_code, filename=safe_filename)

@utils.timer
@mcp.tool()
def create_plotly_chart(
    chart_type: str,
        data: str,
        title: str = "",
        filename: str = "chart.html",
        x_label: str = "",
        y_label: str = "",
        width: int = 800,
        height: int = 600,
) -> str:
    """Create Plotly HTML under artifacts/."""
    try:
        safe_filename = _coerce_artifacts_path(filename)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return _invoke(
        "create_plotly_chart",
        chart_type=chart_type,
        data=data,
        title=title,
        filename=safe_filename,
        x_label=x_label,
        y_label=y_label,
        width=width,
        height=height,
    )

@utils.timer
@mcp.tool()
def save_chart_data(
    data: str, filename: str = "chart_data.json", format: str = "json"
) -> str:
    """Save chart data under artifacts/."""
    try:
        safe_filename = _coerce_artifacts_path(filename)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return _invoke("save_chart_data", data=data, filename=safe_filename, format=format)



if __name__ == "__main__":
    mcp.run(transport="stdio")

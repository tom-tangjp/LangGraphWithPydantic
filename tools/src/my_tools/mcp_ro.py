"""
mcp_ro
Read-only evidence & parsing tools. No network, no shell, no writes.

Transport: stdio (for LangGraph Host spawning this process).
Safety: logs go to stderr; stdout is reserved for MCP protocol.
"""
import sys
import json
import logging
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp_ro")

mcp = FastMCP("mcp_ro")

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
        # FastMCP tool schema here is declared as `-> str` for the wrappers using this helper.
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

def _invoke_list_any(tool_name: str, **kwargs) -> List[Any]:
    """Invoke a LangChain tool (from tools.py) via TOOL_REGISTRY."""
    try:
        tool = _tools.TOOL_REGISTRY[tool_name]
    except Exception as e:
        return [json.dumps({"ok": False, "error": f"tool not found: {tool_name} ({e})"}, ensure_ascii=False)]
    try:
        return tool.invoke(kwargs)
    except Exception as e:
        return [json.dumps({"ok": False, "error": f"invoke {tool_name} failed: {e}"}, ensure_ascii=False)]


@mcp.tool()
def read_file(
    path: str, start: int = 0, max_chars: int = 12000
) -> str:
    """Read-only proxy."""
    return _invoke("read_file", **locals())


@mcp.tool()
def list_dir(
    path: str = ".", max_entries: int = 200
) -> List[str]:
    """Read-only proxy."""
    return _invoke_list_any("list_dir", **locals())


@mcp.tool()
def get_file_info(
    path: str
) -> str:
    """Read-only proxy."""
    return _invoke("get_file_info", **locals())


@mcp.tool()
def read_file_lines(
    path: str, start_line: int = 1, end_line: int = 100
) -> str:
    """Read-only proxy."""
    return _invoke("read_file_lines", **locals())


@mcp.tool()
def walk_dir(
    path: str = ".", max_depth: int = 3, max_files: int = 200
) -> str:
    """Read-only proxy."""
    return _invoke("walk_dir", **locals())


@mcp.tool()
def get_dir_tree(
    path: str = ".", max_depth: int = 5
) -> str:
    """Read-only proxy."""
    return _invoke("get_dir_tree", **locals())


@mcp.tool()
def grep_text(
    pattern: str, path: str = ".", max_matches: int = 50, max_file_size_kb: int = 512
) -> List[Dict[str, Any]]:
    """Read-only proxy."""
    return _invoke_list_any("grep_text", **locals())


@mcp.tool()
def extract_functions(
    path: str
) -> str:
    """Read-only proxy."""
    return _invoke("extract_functions", **locals())


@mcp.tool()
def get_code_metrics(
    path: str = "."
) -> str:
    """Read-only proxy."""
    return _invoke("get_code_metrics", **locals())


@mcp.tool()
def generate_diagram_description(
    description: str, diagram_type: str = "flowchart"
) -> str:
    """Read-only proxy."""
    return _invoke("generate_diagram_description", **locals())


@mcp.tool()
def analyze_data_for_chart(
    data: str, description: str = ""
) -> str:
    """Read-only proxy."""
    return _invoke("analyze_data_for_chart", **locals())


@mcp.tool()
def read_source_file(
    path: str, start: int = 0, max_chars: int = 100*1024*1024
) -> str:
    """Read-only proxy."""
    return _invoke("read_source_file", **locals())


@mcp.tool()
def list_source_files(
    path: str, max_files: int = 500, extensions: Optional[Union[List[str], str]] = None
) -> str:
    """Read-only proxy."""
    return _invoke("list_source_files", **locals())

# @mcp.tool()
# def batch_read_source_files(
#     paths: List[str], max_chars_per_file: int = 50000
# ) -> str:
#     """Read-only proxy."""
#     return _invoke("batch_read_source_files", **locals())


@mcp.tool()
def grep_source_code(
    pattern: str, path: str, max_matches: int = 100, extensions: Optional[Union[List[str], str]] = None
) -> str:
    """Read-only proxy."""
    return _invoke("grep_source_code", **locals())


@mcp.tool()
def extract_cpp_functions(
    path: str
) -> str:
    """Read-only proxy."""
    return _invoke("extract_cpp_functions", **locals())


@mcp.tool()
def get_source_metrics(
    path: str
) -> str:
    """Read-only proxy."""
    return _invoke("get_source_metrics", **locals())

@mcp.tool()
def find_cmake_files(
    path: str
) -> str:
    """Read-only proxy."""
    return _invoke("find_cmake_files", **locals())


@mcp.tool()
def read_compile_commands(
    path: str = "compile_commands.json", json=None
) -> str:
    """Read-only proxy."""
    return _invoke("read_compile_commands", **locals())


@mcp.tool()
def git_status(porcelain: bool = True) -> str:
    """Read-only proxy."""
    return _invoke("git_status", **locals())


@mcp.tool()
def git_diff(paths: List[str] = None, staged: bool = False, max_chars: int = 60000) -> str:
    """Read-only proxy."""
    return _invoke("git_diff", **locals())


@mcp.tool()
def git_show(spec: str = "HEAD", max_chars: int = 60000) -> str:
    """Read-only proxy."""
    return _invoke("git_show", **locals())


if __name__ == "__main__":
    mcp.run(transport="stdio")

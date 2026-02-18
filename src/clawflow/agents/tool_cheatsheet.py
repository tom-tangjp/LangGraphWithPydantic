from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, Union

def _first_line(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s.splitlines()[0].strip()

def _get_pydantic_schema(args_schema: Any) -> Dict[str, Any]:
    if args_schema is None:
        return {}
    # pydantic v2
    if hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            return {}
    # pydantic v1 fallback
    if hasattr(args_schema, "schema"):
        try:
            return args_schema.schema()
        except Exception:
            return {}
    return {}

def _render_params(schema: Dict[str, Any]) -> str:
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    if not props:
        return "—"

    parts = []
    for k, v in props.items():
        t = v.get("type") or v.get("anyOf") or v.get("$ref") or "any"
        # anyOf 太长就压一下
        if isinstance(t, list):
            t = "/".join([str(x.get("type", "any")) if isinstance(x, dict) else "any" for x in t][:3])
        mark = "*" if k in required else ""
        default = ""
        if "default" in v:
            default = f"={v.get('default')!r}"
        parts.append(f"{mark}{k}{default}")
    return ", ".join(parts)

def _iter_tools(tool_map: Union[Dict[str, List[Any]], List[Any]]) -> Iterable[Tuple[str, Any]]:
    """
    Yields: (group_name, tool)
    - If tool_map is dict: key as group
    - If list: group "tools"
    """
    if isinstance(tool_map, dict):
        for g, tools in tool_map.items():
            for t in (tools or []):
                yield str(g), t
    else:
        for t in (tool_map or []):
            yield "tools", t

def render_tool_cheatsheet(
    tool_map: Union[Dict[str, List[Any]], List[Any]],
    *,
    max_tools: int = 60,
    max_chars: int = 3500,
) -> str:
    """
    Deterministic cheatsheet built from actual loaded tools (auto-updates when MCP tools change).
    Uses: tool.name, tool.description, tool.args_schema (if present).
    """
    rows = []
    seen = set()

    for group, tool in _iter_tools(tool_map):
        name = getattr(tool, "name", "") or ""
        if not name or name in seen:
            continue
        seen.add(name)

        desc = _first_line(getattr(tool, "description", "") or "")
        schema = _get_pydantic_schema(getattr(tool, "args_schema", None))
        params = _render_params(schema)

        rows.append((group, name, params, desc))
        if len(rows) >= max_tools:
            break

    # group -> lines
    out_lines: List[str] = []
    current_len = 0

    def emit(line: str):
        nonlocal current_len
        if current_len + len(line) + 1 > max_chars:
            return False
        out_lines.append(line)
        current_len += len(line) + 1
        return True

    rows.sort(key=lambda x: (x[0], x[1]))
    cur_group = None
    for group, name, params, desc in rows:
        if group != cur_group:
            cur_group = group
            if not emit(f"\n### {cur_group}"):
                break
        line = f"- {name}({params})"
        if desc:
            line += f" — {desc}"
        if not emit(line):
            break

    if len(rows) >= max_tools:
        emit("\n(…more tools omitted)")
    return "\n".join(out_lines).strip()

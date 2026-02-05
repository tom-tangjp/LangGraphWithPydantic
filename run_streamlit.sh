#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: activate local virtualenv
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  source "venv/bin/activate"
fi

export AGENT_LOG_LLM_DUMP="${AGENT_LOG_LLM_DUMP:-0}"
export PYTHONIOENCODING=utf-8

# MCP 安全开关：默认尽量保守，按需开启
export MCP_ALLOW_SOURCE_MUTATION="${MCP_ALLOW_SOURCE_MUTATION:-0}"
export MCP_ALLOW_PIP="${MCP_ALLOW_PIP:-0}"
export MCP_ALLOW_PIP_READONLY="${MCP_ALLOW_PIP_READONLY:-0}"
export MCP_ALLOW_PIP_WRITE="${MCP_ALLOW_PIP_WRITE:-0}"
export MCP_ALLOW_PYTHON_MODULES="${MCP_ALLOW_PYTHON_MODULES:-0}"
export MCP_ALLOW_PYTHON_C="${MCP_ALLOW_PYTHON_C:-0}"
export MCP_ALLOW_ALL_COMMANDS="${MCP_ALLOW_ALL_COMMANDS:-0}"
export MAX_TOOL_CALLS_PER_TURN="${MAX_TOOL_CALLS_PER_TURN:-1}"

# WORKSPACE_ROOT / SKILLS_DIR 会被 utils.get_workspace_root()/get_skills_dir() 读取
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SCRIPT_DIR/workspace}"
export SKILLS_DIR="${SKILLS_DIR:-$SCRIPT_DIR/skills}"

mkdir -p "$WORKSPACE_ROOT" || true

nohup streamlit run ui_streamlit.py > streamlit.log 2>&1 &

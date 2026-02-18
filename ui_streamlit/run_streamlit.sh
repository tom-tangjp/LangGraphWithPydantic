#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Optional: activate local virtualenv
if [[ -f "../.venv/bin/activate" ]]; then
  source "../.venv/bin/activate"
elif [[ -f "../venv/bin/activate" ]]; then
  source "../venv/bin/activate"
fi

export AGENT_LOG_LLM_DUMP="${AGENT_LOG_LLM_DUMP:-1}"
export PYTHONIOENCODING=utf-8

# MCP 安全开关：默认尽量保守，按需开启
export MCP_ALLOW_SOURCE_MUTATION="${MCP_ALLOW_SOURCE_MUTATION:-1}"
export MCP_ALLOW_PIP="${MCP_ALLOW_PIP:-1}"
export MCP_ALLOW_PIP_READONLY="${MCP_ALLOW_PIP_READONLY:-1}"
export MCP_ALLOW_PIP_WRITE="${MCP_ALLOW_PIP_WRITE:-1}"
export MCP_ALLOW_PYTHON_MODULES="${MCP_ALLOW_PYTHON_MODULES:-1}"
export MCP_ALLOW_PYTHON_C="${MCP_ALLOW_PYTHON_C:-1}"
export MCP_ALLOW_ALL_COMMANDS="${MCP_ALLOW_ALL_COMMANDS:-1}"
export MAX_TOOL_CALLS_PER_TURN="${MAX_TOOL_CALLS_PER_TURN:-1}"
export MCP_NET_ENABLED="${MCP_NET_ENABLED:-1}"
export MCP_NET_ALLOW_ALL="${MCP_NET_ALLOW_ALL:-1}"
export MCP_NET_ALLOW_DOMAINS="${MCP_NET_ALLOW_DOMAINS:-}"
export MCP_SHELL_ENABLED="${MCP_SHELL_ENABLED:-1}"
export MCP_ALLOW_ANY_PATH="${MCP_ALLOW_ANY_PATH:-1}"
export MEMORY_METRICS_AND_GC="${MEMORY_METRICS_AND_GC:-1}"

# 让 `src.*` 可被直接导入（Streamlit 运行时 cwd 在 ui_streamlit 下）
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# WORKSPACE_ROOT / SKILLS_DIR 会被 utils.get_workspace_root()/get_skills_dir() 读取
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$SCRIPT_DIR/workspace}"
export SKILLS_DIR="${SKILLS_DIR:-$PROJECT_ROOT/src/clawflow/skills/data}"

mkdir -p "$WORKSPACE_ROOT" || true

 python -m pip install -e ../tools

nohup streamlit run ui_streamlit.py > streamlit.log 2>&1 &
#streamlit run ui_streamlit.py

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

export PYTHONIOENCODING=utf-8

# 示例：按需配置（建议用 config.yaml 或环境变量覆盖）
# export WORKSPACE_ROOT="$SCRIPT_DIR"
# export SKILLS_DIR="$SCRIPT_DIR/skills"

# 注意：仓库未内置 app.py，此脚本仅为示例。
chainlit run app.py --port 8000

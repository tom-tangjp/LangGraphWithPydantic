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

export LOG_LLM_CONTENT="${LOG_LLM_CONTENT:-1}"
export SEARCH_BACKEND="${SEARCH_BACKEND:-searxng}"
export SEARXNG_URL="${SEARXNG_URL:-http://localhost:8080}"

WORKSPACE="${WORKSPACE:-$SCRIPT_DIR/tmp}"
LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-$WORKSPACE}"

# Do NOT hardcode API keys in this repo.
# Export it in your shell instead, e.g.:
#   export AGENT_LLM_API_KEY="..."
if [[ -z "${AGENT_LLM_API_KEY:-}" ]]; then
  echo "ERROR: AGENT_LLM_API_KEY is not set" >&2
  exit 1
fi

PYTHONIOENCODING=utf-8 python3 agent_flow_search_not_use_tool_node.py \
  --workspace "$WORKSPACE" \
  --skills-dir ./skills \
  --provider "${AGENT_LLM_PROVIDER:-qwen}" \
  --api-key "$AGENT_LLM_API_KEY" \
  --model "${AGENT_LLM_MODEL:-qwen-plus}" \
  --search-backend "$SEARCH_BACKEND" \
  --base-url "${AGENT_LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

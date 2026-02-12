---
name: python-executor-mcp
description: |
  Execute deterministic Python via an MCP runtime tool for calculations, data processing, and producing artifacts (CSV/PNG/PDF).
Triggers: python, execute python, run script, data analysis, pandas, numpy, matplotlib, report, csv, json, plotting, automation
allowed-tools: python_execute
compatibility: |
  Requires a working MCP Python runtime tool. Update tool name/params if yours differ.
---

# Python Executor via MCP

## Core rules
- Non-interactive only (no input()).
- Print a short summary to stdout; don’t dump huge data.
- Save outputs under `outputs/` and print file paths.
- Validate using `exit_code` and `stderr` (retry with minimal changes).
- No secrets / no destructive ops.

## Tool contract (assumed)
Input: `code`, `timeout_s`, `working_dir`, `capture_output`  
Output: `stdout`, `stderr`, `exit_code`, `artifacts?`, `error?`

## Example (sanity)
```text
python_execute({
  "code": "import sys; print(sys.version)\nprint('ok')\n",
  "timeout_s": 30,
  "capture_output": true,
  "working_dir": null
})

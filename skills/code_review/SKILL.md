---
name: code_review
description: |
  Executable-level code review for Python/LangGraph/LangChain: evidence-first root cause analysis, minimal unified diffs, apply patches, and run local verification. Prefer git_status/git_diff for change evidence to avoid relying on run_bash.
Triggers: code review, bug localization, structured_output parsed=None, schema mismatch, tool calling failure, ToolNode, graph routing, LangGraph, LangChain, Ollama, OpenAI-compatible, regression, patch, diff, pytest, git diff
allowed-tools: repo_scan get_dir_tree list_dir rg_search grep_text read_file_lines read_file git_status git_diff git_show apply_unified_diff patch_apply_and_verify write_text_file check_syntax run_tests format_code analyze_code run_safe_command run_bash web_search web_open http_get
compatibility: Most local tools return JSON strings {"ok": bool, ...}; always check ok/exit_code/stderr. web_search returns a list of results.
---

# SKILL: code_review (with Git tools)

## Goal
- 1–3 root causes with verifiable evidence
- minimal unified diff patch
- runnable verification steps (and run them locally when possible)

## Hard rules (budget + correctness)
- Evidence-first: cite file paths + line ranges (prefer read_file_lines).
- Read the minimal set of files to prove the bug (entrypoint/config/failing module).
- Patch must be minimal; avoid refactors unless required.
- State exactly what was read/applied/verified and what failed.
- Prefer structured tools over run_bash (git_diff/git_status/run_tests).

## Tool strategy (preferred order)
1) Inventory: repo_scan / get_dir_tree / list_dir
2) Change evidence: git_status → git_diff (git_show if needed)
3) Locate: rg_search (preferred) or grep_text
4) Read precisely: read_file_lines (preferred), read_file for larger chunks
5) Patch:
   - Prefer patch_apply_and_verify(diff, ...) for apply + optional format/test
   - Else apply_unified_diff(diff) then check_syntax/run_tests/format_code
6) Verify: check_syntax (mandatory), run_tests (when available), analyze_code/format_code (as needed)
7) If version/docs uncertainty: web_search → web_open (open ≤ 2 pages)

## Triage checklist
- Config/env: wrong provider/model, missing keys, overridden settings
- Structured output: schema too strict, wrong mode for provider, parsing errors
- Tool calling: bind_tools missing, ToolNode not wired, tool_calls not routed
- Graph logic: conditional edges incomplete, termination never reached
- I/O bounds: truncation/max_chars, timeouts, weak error handling

## Output contract
- Root causes (1–3 bullets with evidence)
- Minimal patch (unified diff)
- Changed files
- Verification commands + expected results
- If uncertain: what is missing and why it blocks conclusion

## Minimal example (intent)
```text
git_status(porcelain=true)
git_diff(paths=["src/app.py"], staged=false)

rg_search(query="ToolNode", path="src", max_matches=50)
read_file_lines(path="src/graph.py", start_line=120, end_line=220)

patch_apply_and_verify(diff="...unified diff...", run_tests=true, format=true)

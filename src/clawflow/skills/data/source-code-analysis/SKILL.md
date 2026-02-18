---
name: source_code_analysis
description: |
  Analyze a project codebase pragmatically: identify entrypoints and core modules, summarize key files by directory, map architecture/execution flow, and report bugs/risks with actionable fixes—without token-wasting full-dump reviews.
Triggers: analyze source code, project architecture, execution flow, codebase overview, module summary, find bugs, performance issues, security risks, C++, Python, Java, Go
allowed-tools: list_source_files list_dir read_source_file grep_text walk_dir read_file extract_cpp_functions extract_functions write_text_file
compatibility: Tool names may vary; use available equivalents. Prefer minimal tool usage and graceful fallback if extractors are missing.
---

# Source Code Analysis (Compact)

## Objectives
- Identify entrypoints + core flows.
- Summarize modules by directory and key files.
- Produce architecture + execution-flow overview.
- List concrete bugs/risks + optimizations with references to file paths.

## Hard rules (token/call budget)
- Prioritize **core-path must-read** files first (entrypoints, routing/handlers, config bootstrap, core interfaces, critical data structures).
- Target coverage: read **all core directories**; allow **exclude/skim** for `vendor/`, `third_party/`, `generated/`, `build/`, `dist/`, and large test fixtures.
- Open pages/files efficiently: use `read_file` when available; otherwise read per-file with `max_chars` cap.
- Always report **what was read** vs **skipped** and why. No pretending.

## Minimal workflow
1) Inventory: `list_source_files(root)` + `walk_dir(root)` to understand size and structure.
2) Core-path detection: find entrypoints and routing via `grep_text` for `main`, `cmd`, `server`, `router`, `init`, `Run`, `Start`, etc.
3) Read & extract:
   - Read entrypoints and core modules fully (or first N chars if huge).
   - Use `extract_*` if available; else infer from code + grep.
4) Directory summaries: for each major dir, list responsibilities + key files + interactions.
5) Architecture & flow: provide a simple Mermaid diagram only when confident.
6) Issues: list bugs/risks/optimizations with file+symbol anchors.
7) Output: save report to `project_analysis.md` if `write_text_file` exists; otherwise output in chat.

## Output (short contract)
- Project overview (goal, stack, entrypoints)
- Directory/module map (top-level dirs + roles)
- Key files table (file → role → key symbols)
- Architecture + execution flow (Mermaid optional)
- Issues & fixes (actionable, referenced)
- Stats: total files, analyzed count, skipped rules, date

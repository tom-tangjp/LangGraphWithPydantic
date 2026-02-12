---
name: web-research
description: |
  Web research skill using web_search, web_open, and http_get to find up-to-date information and verify claims with minimal page opens.
Triggers: web search, browse, open url, open link, http get, searxng, tavily, serpapi, bing, google, research, fact check, sources, citations, rag
allowed-tools: web_search web_open http_get
compatibility: Requires outbound network access to the configured search backend and target URLs.
---

# Web Research

## Core rules
- Search first; open only if needed.
- Open pages ≤ 3 per task.
- Default max_chars ≤ 12000; start smaller (8k) and increase only if justified.
- When using web info, return a short **Sources** list (URLs).

## Tool choice
- Discover links → `web_search`
- Read a page → `web_open`
- Fetch API/JSON/plaintext → `http_get`

## Minimal workflow
1) `web_search` (5–10 results)  
2) pick 1–3 credible URLs  
3) `web_open` / `http_get`  
4) answer + Sources

## web_search edge cases
- `title="No Results"` → broaden query / increase recency_days / relax domains.
- `title="[SEARCH_ERROR]"` → retry once with simpler query + smaller max_results; if still failing, report backend/network issue.

## Example
```text
web_search(query="X release notes", recency_days=30, domains=None, max_results=8)
web_open(url="https://...", max_chars=8000, timeout_s=15)

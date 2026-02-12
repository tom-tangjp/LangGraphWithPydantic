---
name: web-research
description: |
  使用 web_search / web_open / http_get 做联网检索与验证：先搜后开、少开精开、控制字符预算，并给出来源链接。
Triggers: 查资料, 搜索, 联网, 打开网页, 读取链接, 抓取接口, 事实核查, 资料来源
allowed-tools: web_search web_open http_get
compatibility: 需要可访问搜索后端与目标 URL 的网络能力。
---

# Web Research（精炼版）

## 核心规则
- 先 `web_search`，确有必要再打开页面。
- 单次任务最多打开网页 ≤ 3。
- `max_chars` 默认 ≤ 12000；优先 8000，不够再增大。
- 使用网页信息时，输出末尾附 **Sources（URL 列表）**。

## 工具选型
- 找链接 → `web_search`
- 读页面 → `web_open`
- 抓 API/JSON/纯文本 → `http_get`

## 最小流程
1) `web_search`（5–10 条）  
2) 选 1–3 条可信链接  
3) `web_open`/`http_get` 读取  
4) 汇总回答 + Sources

## web_search 特殊情况
- `No Results`：放宽 query / 增大 recency_days / 放宽 domains
- `[SEARCH_ERROR]`：用更短 query + 更小 max_results 重试一次；仍失败则说明后端/网络异常

## 示例
```text
web_search(query="X release notes", recency_days=30, domains=None, max_results=8)
web_open(url="https://...", max_chars=8000, timeout_s=15)

---
name: code_review
description: |
  面向 Python/LangGraph/LangChain 的可执行级代码审查：基于证据定位根因，产出最小 unified diff，应用补丁并执行本地验证。优先使用 git_status/git_diff 获取变更证据，减少依赖 run_bash 猜命令。
Triggers: 代码审查, 定位bug, 逻辑解释, structured_output parsed=None, schema不匹配, 工具调用失败, ToolNode, 图路由异常, LangGraph, LangChain, Ollama, OpenAI兼容, 回归, 补丁, diff, pytest, git diff
allowed-tools: repo_scan get_dir_tree list_dir rg_search grep_text read_file_lines read_file git_status git_diff git_show apply_unified_diff patch_apply_and_verify write_text_file check_syntax run_tests format_code analyze_code run_safe_command run_bash web_search web_open http_get
compatibility: 多数本地工具返回 JSON 字符串 {"ok": bool, ...}；必须检查 ok/exit_code/stderr。web_search 返回结果列表。
---

# SKILL: code_review（含 Git 工具）

## 目标
- 输出 1–3 个根因（可复核证据）
- 产出最小补丁（unified diff）
- 给出可运行验证步骤，并尽可能在本地执行验证

## 硬规则（省 token + 保正确）
- 先证据后结论：引用文件路径 + 行号范围（优先 read_file_lines）。
- 只读“证明根因”的最小文件集（入口/配置/相关模块），避免全仓流水账。
- 补丁以最小 diff 为原则；非必要不重构。
- 必须写清：读了哪些、改了哪些、验证了哪些、哪些失败及原因。
- 能不用 run_bash 就不用：优先使用结构化工具（git_diff/git_status/run_tests 等）。

## 工具使用策略（优先级）
1) 结构摸底：repo_scan / get_dir_tree / list_dir
2) 取变更证据：git_status → git_diff（必要时 git_show）
3) 全局定位：rg_search（优先）或 grep_text
4) 精确阅读：read_file_lines（优先），read_file（读大段）
5) 打补丁：
   - 优先 patch_apply_and_verify(diff, ...)：一次性 apply +（可选）format/test
   - 否则 apply_unified_diff(diff) 后手动 check_syntax / run_tests / format_code
6) 验证：
   - 至少做 check_syntax
   - 能跑则 run_tests（pytest）
   - 需要时 analyze_code / format_code
   - 仅在白名单覆盖不到时才用 run_bash
7) 版本/文档不确定：web_search → web_open（打开 ≤ 2 页）

## 快速排查清单（常见根因）
- 配置/环境：provider/model 配置不一致、key 缺失、配置被覆盖
- 结构化输出：schema 太严、provider 不支持该模式、解析异常/字段缺失
- 工具调用链：bind_tools 缺失、ToolNode 未连接、tool_calls 未路由
- 图逻辑：条件边不全、终止条件不收敛、step_idx/状态未更新
- I/O 边界：读取截断、max_chars 不足、超时、错误处理不完善

## 输出约定（必须包含）
- 根因（1–3 点，附证据：文件+行号/日志）
- 最小补丁（unified diff）
- 变更文件列表
- 验证命令 + 期望结果（至少语法检查；能跑就补 pytest）
- 不确定时：缺什么信息、为什么缺它无法下结论

## 最小示例（工具调用意图示例）
```text
# 1) 看工作区状态/变更
git_status(porcelain=true)
git_diff(paths=["src/app.py"], staged=false)

# 2) 定位关键符号并读行号范围
rg_search(query="ToolNode", path="src", max_matches=50)
read_file_lines(path="src/graph.py", start_line=120, end_line=220)

# 3) 应用最小 diff 并验证
patch_apply_and_verify(diff="...unified diff...", run_tests=true, format=true)

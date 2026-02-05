# SKILL: code_review

## 一句话简介
对 Python/LangGraph/LangChain 项目进行可执行层面的代码审查：定位 bug、结构化输出失败、tool 调用链路/状态机错误，并给出最小可落地 patch 与验证步骤。

---

## 适用场景（什么时候用）
- 用户要求 review 代码、定位 bug、解释逻辑、提升稳定性/可维护性。
- structured_output 解析失败（parsed=None / schema 不兼容）、工具调用不生效、graph 路由不符合预期。
- 多模型/多 provider 适配（Ollama / OpenAI-compatible / 各厂商 API）或配置化改造。

---

## 不适用场景（不要用）
- 纯写作/润色且无需读代码。
- 用户未提供任何源码/日志，也无法访问仓库且问题不含可推断信息（此时先向用户索要最小复现）。

---

## 依赖的工具（由宿主系统提供）
最小可用工具集（名称可映射）：
- `list_dir(path) -> entries[]`：快速了解目录结构
- `read_text_file(path, start?, max_chars?) -> text`：阅读源码/配置
- `grep_text(pattern, path?, max_matches?) -> matches[]`：全局检索关键符号/错误信息

可选增强工具：
- `web_search(...)` / `web_open(...)`：查 LangChain/LangGraph 版本差异、第三方 API 文档

---

## 标准流程（建议）
### Step 1：建立问题边界（最小复现）
- 明确：入口脚本/触发方式、期望行为、实际行为、报错堆栈、依赖版本。
- 若信息不足：优先要求用户提供「报错堆栈 + 关键文件 + 运行命令」。

### Step 2：结构化检查清单（按优先级）
1) **运行时配置**：env/配置文件是否被覆盖、路径 root 是否被意外重写。
2) **LLM 适配**：
   - provider 是否匹配（Ollama vs OpenAI-compatible）。
   - structured 输出方法是否被 provider 支持（json_schema / function_calling / json_mode）。
3) **工具调用链路**：
   - 模型是否 bind_tools；ToolNode 是否连通；tool_calls 是否被正确路由。
4) **Graph/状态机**：
   - 条件边是否覆盖所有分支；done/step_idx 是否能收敛。
5) **I/O 与安全**：
   - 文件读取是否允许越界；网络抓取是否有超时/重试；输出是否过长导致截断。

### Step 3：给出最小 patch
- 只改动必要位置（避免大规模重构）。
- 补充：日志、异常兜底、参数校验、合理默认值。

### Step 4：验证策略
- 给出一组可直接执行的验证命令。
- 对关键 bug，至少给出一个「失败用例」和「修复后预期」。

---

## 输出要求
- 先给结论：问题根因（1~3 点）
- 再给行动：最小 patch（diff 或关键代码段）+ 验证步骤
- 若存在不确定：明确缺少什么信息会影响结论

# LangGraphWithPydantic

一个基于 `LangGraph` + `Pydantic` 的多智能体（Intent/Planner/Agent/Reflect/Respond）示例工程，内置 **MCP（Model Context Protocol）工具适配层**，并提供 `Streamlit` 调试 UI。

## 功能概览

- 多角色闭环：`intent → initial_plan → (agent ↔ tools ↔ reflect)* → respond`（见 `src/clawflow/agents/llm.py`）
- 结构化输出：`IntentModel / PlanModel / ReflectModel` 全部走 Pydantic schema（见 `src/clawflow/agents/agent.py`、`src/clawflow/agents/llm.py`）
- 工具隔离：工具通过 MCP 子进程暴露，分为只读、受控执行、产物写入（见 `tools/src/my_tools/mcp_ro.py`、`tools/src/my_tools/mcp_exec_safe.py`、`tools/src/my_tools/mcp_artifacts.py`）
- 安全兜底：工具输出做截断、连续工具调用熔断、shell 默认禁用（见 `src/clawflow/agents/llm.py`、`tools/src/my_tools/mcp_shell.py`）
- 可视化调试：Streamlit UI 展示步骤进度、日志、Token 统计、Mermaid 时序图（见 `ui_streamlit/ui_streamlit.py`、`ui_streamlit/trace_visualizer.py`）

## 目录结构（核心文件）

- `src/clawflow/main.py`：命令行入口，构建图并运行一次请求
- `src/clawflow/agents/llm.py`：LLM 构建、LangGraph 编排、MCP 工具初始化、反思/响应节点
- `src/clawflow/agents/agent.py`：状态与结构化 schema（Intent/Plan/Reflect/State）
- `src/clawflow/skills/`：技能加载与内置技能数据（默认技能在 `src/clawflow/skills/data/`）
- `ui_streamlit/ui_streamlit.py`：Streamlit 调试 UI（流式运行图、展示 traces/步骤/Token）
- `ui_streamlit/trace_visualizer.py`：把 traces 转成 Mermaid 时序图
- `tools/src/my_tools/`：独立的 `my_tools` 包（MCP 服务端、工具实现、Web 工具）
  - `tools/src/my_tools/tools.py`：工具实现与注册（文件/grep/git/受控执行/patch 等）
  - `tools/src/my_tools/mcp_*.py`：按权限拆分的 MCP 服务端
  - `tools/src/my_tools/web_tools.py`：网络检索/抓取工具（按 `SEARCH_BACKEND/SEARXNG_URL` 等选择后端）

## 环境要求

- Python `3.11+`
- 可选：Docker（启动 `searxng` 搜索后端 / 容器化运行）

依赖见 `requirements.txt`。

## 安装

```bash
pip install -r requirements.txt
pip install -e tools
```

说明：`my_tools` 是一个独立包（位于 `tools/`），`src/clawflow/agents/llm.py` 默认会通过 `my_tools.mcp_*` 启动 MCP 子进程，因此需要先把它装到当前 Python 环境里。

## 快速开始（推荐）

1) 准备配置：

```bash
cp config_example/config.yaml config.yaml
```

2) 编辑 `config.yaml`（至少设置工作区、技能目录、四个角色模型与 provider 连接信息）：

- `workspace.root`：工具读写/扫描的根目录（例如 `./ui_streamlit/workspace`）
- `skills.dir`：技能目录（内置技能默认在 `./src/clawflow/skills/data`）
- `intent/planner/agent/reflector`：每个角色的 `provider`/`model`
- `llm.providers.<provider>.base_url/api_key`：对应 provider 的连接信息（`ollama` 通常不需要 `api_key`）
- `search.backend`：搜索后端（`searxng | tavily | serpapi | bing | none`）
- `search.url`：当 `search.backend=searxng` 时使用的 SearxNG URL（注意：代码读取的是 `search.url`，不是 `searxng.url`）

3) 启动（先用 CLI 验证一轮，再用 UI 调试）：

```bash
python -m src.clawflow.main --question "用一句话解释这个项目是做什么的"
streamlit run ui_streamlit/ui_streamlit.py
```

## 配置说明（环境变量 / 配置文件）

### 配置读取优先级

整体优先级可理解为：

1) 同名“直读环境变量”（例如代码里读 `WORKSPACE.ROOT` / `INTENT.PROVIDER` 这类 key；一般用于 Docker Compose 等场景）
2) `AGENT_*` 环境变量（推荐在本地 shell 用这种方式覆盖；会被映射到配置的层级结构）
3) `config.yaml`（仓库根目录；优先级高于 `.env`）
4) 默认值

说明：`.env` 仅支持最简单的 `key=value` 读取，且不支持层级结构；建议以 `config.yaml` 为主。

### 必需配置（最小可跑）

`ui_streamlit/ui_streamlit.py` 启动会校验：

- `WORKSPACE.ROOT`（或通过 `AGENT_WORKSPACE_ROOT` 映射得到）
- `SKILLS.DIR`（或通过 `AGENT_SKILLS_DIR` 映射得到）
- `SEARCH.BACKEND`（或通过 `AGENT_SEARCH_BACKEND` 映射得到）
- `INTENT.PROVIDER` / `INTENT.MODEL`（或 `AGENT_INTENT_PROVIDER` / `AGENT_INTENT_MODEL`）
- `PLANNER.PROVIDER` / `PLANNER.MODEL`（或 `AGENT_PLANNER_PROVIDER` / `AGENT_PLANNER_MODEL`）
- `AGENT.PROVIDER` / `AGENT.MODEL`（或 `AGENT_AGENT_PROVIDER` / `AGENT_AGENT_MODEL`）

另外，构建图时还需要 `REFLECTOR.PROVIDER` / `REFLECTOR.MODEL`（见 `src/clawflow/agents/llm.py` 的 `build_llm()`）。

### 关键配置项（对应 `config_example/config.yaml`）

- LLM Provider 连接信息：
  - `llm.providers.qwen.base_url` / `llm.providers.qwen.api_key`
  - `llm.providers.doubao.base_url` / `llm.providers.doubao.api_key`
  - `llm.providers.openai_compat.base_url` / `llm.providers.openai_compat.api_key`
  - `llm.providers.ollama.base_url`（本地 Ollama）
- 工作空间与技能：
  - `workspace.root`
  - `skills.dir`
- 搜索：
  - `search.backend`：`searxng | tavily | serpapi | bing | none`
  - `search.url`：当 backend=searxng（会被写入运行时环境变量 `SEARXNG_URL` 供 `my_tools` 使用）
  - `tavily.api_key` / `serpapi.api_key` / `bing.api_key`：当 backend=对应服务
- 日志：
  - `log.level`：如 `INFO`
  - `log.llm.content`：是否记录 LLM 调用内容（建议只在本地调试开启）

### 环境变量常用清单

如果你不想改配置文件，也可以用环境变量覆盖：

- 推荐（本地 shell）：使用 `AGENT_*` 前缀覆盖层级配置，例如：
  - `AGENT_WORKSPACE_ROOT`、`AGENT_SKILLS_DIR`
  - `AGENT_SEARCH_BACKEND`、`AGENT_SEARCH_URL`
  - `AGENT_INTENT_PROVIDER` / `AGENT_INTENT_MODEL`
  - `AGENT_PLANNER_PROVIDER` / `AGENT_PLANNER_MODEL`
  - `AGENT_AGENT_PROVIDER` / `AGENT_AGENT_MODEL`
  - `AGENT_REFLECTOR_PROVIDER` / `AGENT_REFLECTOR_MODEL`
- Docker/Compose：可以直接设置代码读取的同名 key（例如 `WORKSPACE.ROOT`、`INTENT.PROVIDER`），但这类带点号的变量名不适合在普通 shell 里 `export`。
- 调试：`AGENT_LOG_LLM_DUMP=1`、`LOG.LLM.CONTENT=1`、`MAX_TOOL_CALLS_PER_TURN=1`

安全提醒：请勿把真实 API Key 写入并提交到仓库。

## 运行方式

### 1) 命令行（单次请求）

```bash
python -m src.clawflow.main --question "用一句话解释这个项目是做什么的"
```

说明：`src/clawflow/main.py` 本身是“驱动 LangGraph 跑一轮”的 CLI 包装。最稳妥的方式还是把模型与 provider 连接信息写入 `config.yaml`，然后直接运行。

（如果你希望用命令行/环境变量覆盖配置：本地 shell 推荐用 `AGENT_*` 环境变量覆盖角色 provider/model（如 `AGENT_INTENT_PROVIDER`、`AGENT_PLANNER_MODEL`）；provider 连接信息（`llm.providers.*` 的 `base_url/api_key`）建议仍写在 `config.yaml`。）

### 2) Streamlit UI（推荐调试）

```bash
streamlit run ui_streamlit/ui_streamlit.py
```

UI 启动时会校验关键配置：`WORKSPACE.ROOT / SKILLS.DIR / SEARCH.BACKEND / INTENT.* / PLANNER.* / AGENT.*`（见 `ui_streamlit/ui_streamlit.py`）。

### 3) 启动搜索依赖（可选：SearxNG）

当 `SEARCH.BACKEND=searxng` 时，需要一个可用的 SearxNG 服务。

方式 A：使用本仓库根目录的 `docker-compose.yml` 只启动 searxng（最简单）

```bash
docker compose up -d searxng
```

方式 B：使用 `searxng-docker/` 目录启动完整套件（含 caddy / valkey）

```bash
cd searxng-docker
docker compose up -d
```

启动后将 `AGENT_SEARCH_URL`（或 `search.url`）指向 `http://localhost:8080`（默认端口）。

### 4) Docker Compose（可选：容器化跑 UI + SearxNG）

```bash
docker compose up --build
```

- 端口：`8080`（SearxNG）、`8501`（Streamlit，需自行切换容器启动命令）
- 说明：当前 `Dockerfile` 默认 CMD 指向 `chainlit run app.py`，但仓库里未提供 `app.py`；如需容器化运行，请自行调整启动命令（例如改为 `streamlit run ui_streamlit/ui_streamlit.py`）。

## 运行机制（简述）

- `intent`：将用户请求解析为结构化意图（严格 JSON schema，禁止工具调用）
- `initial_plan/dynamic_plan`：生成（或动态追加）可执行步骤，步骤角色包括 `researcher/solver/writer/code_researcher/code_reviewer`
- `agent_node`：按 step 执行，允许产生 tool_calls
- `safe_tool_node`：执行工具并对输出做强截断，避免 token 爆炸（见 `src/clawflow/agents/llm.py`）
- `reflect`：按验收标准决定 `accept/retry/revise_plan/finish/generate_next_step`
- `respond`：汇总 artifacts 输出最终答复（禁止工具）

## MCP 与 Tools 说明

本项目把“工具能力”拆成两层：

1) `tools/src/my_tools/tools.py`：定义本地 LangChain `@tool`（文件/代码/命令/可视化等）
2) `tools/src/my_tools/mcp_*.py`：把 `tools/src/my_tools/tools.py` 里的工具按权限分组，通过 MCP（stdio 子进程）暴露给 LangGraph

LangGraph 侧在 `src/clawflow/agents/llm.py` 的 `init_mcp_tools()` 中启动子进程并拉取 tool schema，然后把工具按角色聚合供节点分发。

### 工具与服务端（概览）

- 只读：`tools/src/my_tools/mcp_ro.py`（例如 `read_file`、`list_dir`、`grep_text`、`git_diff` 等）
- 受控执行：`tools/src/my_tools/mcp_exec_safe.py`（例如 `run_safe_command`、`run_tests`、`format_code` 等，受开关控制）
- 产物写入：`tools/src/my_tools/mcp_artifacts.py`（例如 `write_text_file`、`save_mermaid_diagram` 等）
- Shell（危险）：`tools/src/my_tools/mcp_shell.py`（`run_bash`，默认禁用）
- 网络（可选）：`tools/src/my_tools/mcp_net.py` 与 `tools/src/my_tools/web_tools.py`（`web_search/web_open/http_get`）

完整工具名以 `tools/src/my_tools/tools.py` / `tools/src/my_tools/web_tools.py` 中的 `@tool("...")` 声明为准。

### 角色可用工具集

`src/clawflow/agents/llm.py` 的 `init_mcp_tools()` 会把工具组合成不同权限集合（高层含义如下）：

- `researcher`：只读 + 网络（检索/阅读/grep/代码扫描）
- `solver`：只读 + 受控执行 + 网络（允许 pytest/格式化等受控动作，受开关限制）
- `writer`：只读 + 产物写入 + 网络（把最终交付物写入 `artifacts/`）
- `code_reviewer`：只读（审阅模式）

### 如何新增/扩展工具

1) 在 `tools/src/my_tools/tools.py` 里新增一个 `@tool("...")` 方法并实现逻辑
2) 在对应的 `tools/src/my_tools/mcp_ro.py` / `tools/src/my_tools/mcp_exec_safe.py` / `tools/src/my_tools/mcp_artifacts.py` 中增加一个同名 proxy（调用 `TOOL_REGISTRY[tool_name].invoke(...)`）
3) 需要的话，在 `src/clawflow/agents/llm.py:init_mcp_tools()` 的角色分组里把新工具纳入对应角色

## 测试

```bash
pytest -q
```

也可以运行内置的“未来日期处理”回归脚本：`scripts/run_test.sh`（会用 `RUNTIME_UTC_OVERRIDE` 模拟时间，相关逻辑见 `src/clawflow/utils/utils.py`）。

## 备注（安全相关）

- `tools/src/my_tools/mcp_shell.py` 提供任意 shell 执行能力，但默认通过 `MCP_SHELL_ENABLED=0` 禁用；如需开启请自行评估风险。
- `tools/src/my_tools/mcp_exec_safe.py` 默认只允许白名单命令；写入类动作（format/pip install 等）有额外开关约束。

### MCP 安全开关速查

- Shell：`MCP_SHELL_ENABLED=1`（可选 `MCP_SHELL_ALLOW_PREFIXES=git ,pytest` 限制前缀）
- 受控执行：
  - `MCP_ALLOW_SOURCE_MUTATION=1`（允许 format 类写入）
  - `MCP_ALLOW_PIP_READONLY=1` / `MCP_ALLOW_PIP_WRITE=1`
  - `MCP_ALLOW_PYTHON_MODULES=1` / `MCP_ALLOW_PYTHON_C=1`
  - `MCP_ALLOW_ALL_COMMANDS=1`（不建议开启）
- 网络：`MCP_NET_ENABLED=1`，并用 `MCP_NET_ALLOW_DOMAINS=example.com,github.com` 或 `MCP_NET_ALLOW_ALL=1` 控制放行

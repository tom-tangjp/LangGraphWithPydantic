# LangGraphWithPydantic

一个基于 `LangGraph` + `Pydantic` 的多智能体（Intent/Planner/Agent/Reflect/Respond）示例工程，内置 **MCP（Model Context Protocol）工具适配层**，并提供 `Streamlit` 调试 UI。

## 功能概览

- 多角色闭环：`intent → initial_plan → (agent ↔ tools ↔ reflect)* → respond`（见 `llm.py`）
- 结构化输出：`IntentModel / PlanModel / ReflectModel` 全部走 Pydantic schema（见 `agent.py`、`llm.py`）
- 工具隔离：工具通过 MCP 子进程暴露，分为只读、受控执行、产物写入（见 `mcp_ro.py`、`mcp_exec_safe.py`、`mcp_artifacts.py`）
- 安全兜底：工具输出做截断、连续工具调用熔断、shell 默认禁用（见 `llm.py`、`mcp_shell.py`）
- 可视化调试：Streamlit UI 展示步骤进度、日志、Token 统计、Mermaid 时序图（见 `ui_streamlit.py`、`trace_visualizer.py`）

## 目录结构（核心文件）

- `main.py`：命令行入口，构建图并运行一次请求
- `ui_streamlit.py`：Streamlit 调试 UI（流式运行图、展示 traces/步骤/Token）
- `llm.py`：LLM 构建、LangGraph 编排、ToolNode 安全执行、Reflection/Respond 节点
- `agent.py`：状态与结构化 schema（Intent/Plan/Reflect/State）
- `tools.py`：本地工具实现（读写文件、grep、运行受控命令、代码度量/图表等）
- `mcp_adapter.py`：把 MCP 工具转成 LangChain `StructuredTool`，并管理子进程 session
- `mcp_ro.py` / `mcp_exec_safe.py` / `mcp_artifacts.py`：MCP 服务端（只读/受控执行/写产物）
- `web_tools.py`：Web 搜索与抓取（支持 `searxng/tavily/serpapi/bing/none`）
- `skills/`：技能文档（`skills_registry.py` 会扫描并提供 `skills_list/skills_load` 工具）

## 环境要求

- Python `3.11+`
- 可选：Docker（启动 `searxng` 搜索后端 / 容器化运行）

依赖见 `requirements.txt`。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始（推荐）

1) 准备配置：

```bash
cp config_example/config.yaml config.yaml
```

2) 编辑 `config.yaml`（至少设置工作区、技能目录、四个角色模型与 provider 连接信息）：

- `workspace.root`：工具读写/扫描的根目录
- `skills.dir`：技能目录（默认 `./skills`）
- `intent/planner/agent/reflector`：每个角色的 `provider`/`model`
- `llm.providers.<provider>.base_url/api_key`：对应 provider 的连接信息（`ollama` 通常不需要 `api_key`）

3) 启动（先用 CLI 验证一轮，再用 UI 调试）：

```bash
python main.py --question "用一句话解释这个项目是做什么的"
streamlit run ui_streamlit.py
```

## 配置说明（环境变量 / 配置文件）

### 配置读取优先级

整体优先级可理解为：

1) 同名环境变量（如 `WORKSPACE.ROOT`）
2) `config.yaml/config.yml/config.json`（按顺序找到第一个就用）
3) 默认值

说明：代码里也支持读取 `.env`，但它是“简易 key=value 读取”，不建议作为主配置来源（推荐统一写到 `config.yaml`）。

### 必需配置（最小可跑）

`ui_streamlit.py` 启动会校验：

- `WORKSPACE.ROOT`
- `SKILLS.DIR`
- `SEARCH.BACKEND`
- `INTENT.PROVIDER` / `INTENT.MODEL`
- `PLANNER.PROVIDER` / `PLANNER.MODEL`
- `AGENT.PROVIDER` / `AGENT.MODEL`

另外，构建图时还需要 `REFLECTOR.PROVIDER` / `REFLECTOR.MODEL`（见 `llm.py` 的 `build_llm()`）。

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
  - `searxng.url`：当 backend=searxng
  - `tavily.api_key` / `serpapi.api_key` / `bing.api_key`：当 backend=对应服务
- 日志：
  - `log.level`：如 `INFO`
  - `log.llm.content`：是否记录 LLM 调用内容（建议只在本地调试开启）

### 环境变量常用清单

如果你不想改配置文件，也可以用环境变量覆盖（推荐在 Docker / CI / 临时调试时使用）：

- 运行基础：`WORKSPACE.ROOT`、`SKILLS.DIR`、`SEARCH.BACKEND`、`LOG.LEVEL`
- 角色模型：`INTENT.* / PLANNER.* / AGENT.* / REFLECTOR.*`（例如 `INTENT.PROVIDER=ollama`、`INTENT.MODEL=qwen3:32b`）
- Provider 连接：`LLM.PROVIDERS.<PROVIDER>.BASE_URL`、`LLM.PROVIDERS.<PROVIDER>.API_KEY`
- 搜索：`SEARXNG.URL`、`TAVILY.API_KEY`、`SERPAPI.API_KEY`、`BING.API_KEY`
- 调试：`AGENT_LOG_LLM_DUMP=1`、`LOG.LLM.CONTENT=1`、`MAX_TOOL_CALLS_PER_TURN=1`

安全提醒：仓库里的 `run.sh` / `run_streamlit.sh` 是本地脚本示例，包含绝对路径与敏感开关示例；请勿把真实 API Key 写入并提交到仓库。

## 运行方式

### 1) 命令行（单次请求）

```bash
python main.py --question "用一句话解释这个项目是做什么的"
```

说明：`main.py` 本身是“驱动 LangGraph 跑一轮”的 CLI 包装。最稳妥的方式还是把模型与 provider 连接信息写入 `config.yaml`，然后直接运行。

（如果你希望完全靠命令行覆盖配置：请优先用环境变量覆盖 `INTENT.* / PLANNER.* / AGENT.* / REFLECTOR.*` 与 `LLM.PROVIDERS.*`，因为 LLM 初始化读取的是这些 key。）

### 2) Streamlit UI（推荐调试）

```bash
streamlit run ui_streamlit.py
```

UI 启动时会校验关键配置：`WORKSPACE.ROOT / SKILLS.DIR / SEARCH.BACKEND / INTENT.* / PLANNER.* / AGENT.*`（见 `ui_streamlit.py`）。

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

启动后将 `SEARXNG.URL`（或 `searxng.url`）指向 `http://localhost:8080`（默认端口）。

### 4) Docker Compose（可选：容器化跑 UI + SearxNG）

```bash
docker compose up --build
```

- 端口：`8080`（SearxNG）、`8501`（Streamlit，需自行切换容器启动命令）
- 说明：当前 `Dockerfile` 默认 CMD 指向 `chainlit run app.py`，但仓库里未提供 `app.py`。如果你只想跑 Streamlit，建议在本机直接 `streamlit run ui_streamlit.py`，或自行改 Docker 启动命令。

## 运行机制（简述）

- `intent`：将用户请求解析为结构化意图（严格 JSON schema，禁止工具调用）
- `initial_plan/dynamic_plan`：生成（或动态追加）可执行步骤，步骤角色包括 `researcher/solver/writer/code_researcher/code_reviewer`
- `agent_node`：按 step 执行，允许产生 tool_calls
- `safe_tool_node`：执行工具并对输出做强截断，避免 token 爆炸（见 `llm.py:863`）
- `reflect`：按验收标准决定 `accept/retry/revise_plan/finish/generate_next_step`
- `respond`：汇总 artifacts 输出最终答复（禁止工具）

## MCP 与 Tools 说明

本项目把“工具能力”拆成两层：

1) `tools.py`：定义本地 LangChain `@tool`（文件/代码/命令/可视化等）
2) `mcp_*.py`：把 `tools.py` 里的工具按权限分组，通过 MCP（stdio 子进程）暴露给 LangGraph

LangGraph 侧通过 `mcp_adapter.py` 启动子进程并拉取 tool schema，`llm.py` 的 `init_mcp_tools()` 会把工具按角色聚合并注册到全局 `TOOL_REGISTRY` 供 `safe_tool_node` 分发。

### 工具清单（按 MCP 服务端分组）

**1) 只读工具（`mcp_ro.py`）**：不执行外部命令、不联网、不写文件，主要用于读取与分析。

- 文件/目录：`read_file`、`read_file_lines`、`list_dir`、`walk_dir`、`get_dir_tree`、`get_file_info`
- 搜索：`grep_text`、`grep_source_code`
- 源码分析：`read_source_file`、`list_source_files`、`extract_functions`、`get_code_metrics`
- C/C++：`extract_cpp_functions`、`get_source_metrics`、`find_cmake_files`、`read_compile_commands`
- Git：`git_status`、`git_diff`、`git_show`
- 可视化/数据：`generate_diagram_description`、`analyze_data_for_chart`

**2) 受控执行工具（`mcp_exec_safe.py`）**：通过白名单与开关控制“可执行动作”。

- 命令执行（白名单）：`run_safe_command`
- 代码质量/测试：`analyze_code`、`run_tests`、`check_syntax`
- C/C++：`run_cpp_linter`、`check_cpp_syntax`
- 格式化（需要显式允许写入）：`format_code`、`format_cpp`

**3) 产物写入工具（`mcp_artifacts.py`）**：只允许写到 `artifacts/`（或 `MCP_ARTIFACTS_PREFIX`）下。

- 写目录/文件：`ensure_dir`、`write_text_file`
- 图表：`save_mermaid_diagram`、`create_plotly_chart`、`save_chart_data`

**4) 危险 Shell 工具（`mcp_shell.py`）**：`run_bash`（默认禁用）。

**5) 网络工具（可选，`mcp_net.py`）**：提供带域名 allowlist 的 `web_search/web_open/http_get`（默认禁用）。

另外，项目也内置了 Python 版网络工具 `web_tools.py`：`web_search`、`web_open`、`http_get`，由 `SEARCH.BACKEND` 选择后端（`searxng|tavily|serpapi|bing|none`）。

### 角色可用工具集

`llm.py` 的 `init_mcp_tools()` 会把工具组合成不同权限集合（高层含义如下）：

- `researcher`：只读 + 网络（检索/阅读/grep/代码扫描）
- `solver`：只读 + 受控执行 + 网络（允许 pytest/格式化等受控动作，受开关限制）
- `writer`：只读 + 产物写入 + 网络（把最终交付物写入 `artifacts/`）
- `code_reviewer`：只读（审阅模式）

### 如何新增/扩展工具

1) 在 `tools.py` 里新增一个 `@tool("...")` 方法并实现逻辑
2) 在对应的 `mcp_ro.py` / `mcp_exec_safe.py` / `mcp_artifacts.py` 中增加一个同名 proxy（调用 `TOOL_REGISTRY[tool_name].invoke(...)`）
3) 需要的话，在 `llm.py:init_mcp_tools()` 的角色分组里把新工具纳入对应角色

## 测试

```bash
pytest -q
```

也可以运行内置的“未来日期处理”回归脚本：`run_test.sh`（会用 `RUNTIME_UTC_OVERRIDE` 模拟时间，见 `utils.py:663`）。

## 备注（安全相关）

- `mcp_shell.py` 提供任意 shell 执行能力，但默认通过 `MCP_SHELL_ENABLED=0` 禁用；如需开启请自行评估风险。
- `mcp_exec_safe.py` 默认只允许白名单命令；写入类动作（format/pip install 等）有额外开关约束。

### MCP 安全开关速查

- Shell：`MCP_SHELL_ENABLED=1`（可选 `MCP_SHELL_ALLOW_PREFIXES=git ,pytest` 限制前缀）
- 受控执行：
  - `MCP_ALLOW_SOURCE_MUTATION=1`（允许 format 类写入）
  - `MCP_ALLOW_PIP_READONLY=1` / `MCP_ALLOW_PIP_WRITE=1`
  - `MCP_ALLOW_PYTHON_MODULES=1` / `MCP_ALLOW_PYTHON_C=1`
  - `MCP_ALLOW_ALL_COMMANDS=1`（不建议开启）
- 网络：`MCP_NET_ENABLED=1`，并用 `MCP_NET_ALLOW_DOMAINS=example.com,github.com` 或 `MCP_NET_ALLOW_ALL=1` 控制放行

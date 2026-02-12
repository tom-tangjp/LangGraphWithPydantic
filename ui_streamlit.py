import asyncio
import html
import logging
import time
import uuid
import streamlit as st
import streamlit.components.v1 as components

import utils

# 你现有的构建函数（按你的项目改 import）
from llm import build_llm, build_reflection_multi_agent_graph
from tools import TOOL_REGISTRY
from trace_visualizer import generate_mermaid_sequence
from mcp_adapter import CLIENT_MANAGER

import os, gc, asyncio, tracemalloc
import psutil

tracemalloc.start(25)
_proc = psutil.Process(os.getpid())

def mem_snapshot(tag: str):
    rss = _proc.memory_info().rss / (1024 * 1024)
    cur, peak = tracemalloc.get_traced_memory()
    cur /= 1024 * 1024
    peak /= 1024 * 1024
    try:
        loop = asyncio.get_running_loop()
        tasks = len(asyncio.all_tasks(loop))
    except Exception:
        tasks = -1
    print(f"[MEM] {tag} rss={rss:.1f}MB py_cur={cur:.1f}MB py_peak={peak:.1f}MB gc={gc.get_count()} tasks={tasks}")

def render_mermaid(code: str, height=600):
    """
    使用 HTML/JS 渲染 Mermaid 图表
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
    </head>
    <body>
        <div class="mermaid">
            {code}
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=True)


def reconstruct_state_from_traces(traces):
    """从 traces 中重建当前状态"""
    current_state = {
        "plan": None,
        "step_idx": 0,
        "artifacts": {},
        # 记录同一步骤的多次产出（重试/工具返回/多轮对话）
        # step_id -> [ {"trace_i": int, "node": str, "artifact": dict} ]
        "artifact_history": {},
        "step_failures": {},
        "step_tool_stats": {},
        "no_progress": {},
        "last_feedback": {},
        "executed_steps": [],
        "pending_steps": [],
        "last_step_id": None,
        "last_agent_role": None,
        "iter_count": 0,
        "done": False,
    }

    # 按时间顺序处理 traces（从旧到新），模拟状态更新
    for trace_i, trace in enumerate(traces):
        patch = trace.get("patch") or {}
        node = trace.get("node", "")

        # 调试：打印节点和补丁结构
        # print(f"节点: {node}, 补丁键: {list(patch.keys())}")

        # 应用补丁到当前状态
        # LangGraph 的补丁可能是嵌套的，例如 {"plan": {...}} 或 {"step_idx": 1}
        # 但也可能是更复杂的结构，例如 {"artifacts": {"step1": {...}}}
        for key, value in patch.items():
            if key in [
                "plan",
                "step_idx",
                "iter_count",
                "done",
                "executed_steps",
                "pending_steps",
                "last_step_id",
                "last_agent_role",
            ]:
                current_state[key] = value
            elif key in [
                "artifacts",
                "step_failures",
                "step_tool_stats",
                "no_progress",
                "last_feedback",
            ]:
                if isinstance(value, dict):
                    # 如果是字典，合并更新
                    if key == "artifacts":
                        # artifacts 可能包含多个步骤的输出，需要合并
                        for step_id, artifact in value.items():
                            current_state[key][step_id] = artifact
                            # 同步累计历史，便于 UI 展示“看起来执行了很多轮”
                            if step_id not in current_state["artifact_history"]:
                                current_state["artifact_history"][step_id] = []
                            current_state["artifact_history"][step_id].append(
                                {
                                    "trace_i": trace_i,
                                    "node": node,
                                    "artifact": artifact,
                                }
                            )
                    else:
                        current_state[key].update(value)

    return current_state


def _safe_repr(val, max_len: int = 200) -> str:
    try:
        s = repr(val)
    except Exception:
        s = f"<{type(val).__name__} repr_failed>"
    if max_len and len(s) > max_len:
        return s[:max_len] + "…"
    return s


def _normalize_steps(steps):
    """Normalize plan steps to `list[dict]` for UI rendering.

    Upstream plan generation may occasionally return malformed structures
    (e.g., steps is a string, or step items are not dict-like). The UI should
    be resilient and never crash while rendering.
    """

    if steps is None:
        return []

    # Single step dict
    if isinstance(steps, dict):
        return [steps]

    # A common failure mode: steps accidentally becomes a string
    if isinstance(steps, str):
        logging.warning(
            "ui.normalize_steps: steps is str; coercing to single step | steps=%s",
            _safe_repr(steps),
        )
        return [{"id": "step1", "agent": "未知", "task": steps}]

    # Anything else that isn't a list/tuple: wrap as a single step
    if not isinstance(steps, (list, tuple)):
        logging.warning(
            "ui.normalize_steps: steps is %s; coercing to single step | steps=%s",
            type(steps).__name__,
            _safe_repr(steps),
        )
        return [{"id": "step1", "agent": "未知", "task": str(steps)}]

    out = []
    for i, s in enumerate(steps):
        if isinstance(s, dict):
            out.append(s)
            continue
        if hasattr(s, "model_dump"):
            try:
                out.append(s.model_dump())
                continue
            except Exception:
                # fall through to coercion
                pass

        logging.warning(
            "ui.normalize_steps: step[%s] is %s; coercing to dict | step=%s",
            i,
            type(s).__name__,
            _safe_repr(s),
        )
        out.append({"id": f"step{i+1}", "agent": "未知", "task": str(s)})

    return out


def _escape_md(s: str) -> str:
    return html.escape(str(s or ""))


def _render_step_details_html(
    *,
    step_idx: int,
    step: dict,
    artifact: dict,
    history: list,
    status: str,
    failure_count: int,
    no_progress: bool,
    feedback: dict,
    max_output_chars: int = 8000,
) -> str:
    """Render a single step as <details> HTML block."""
    step_id = str(step.get("id") or f"step{step_idx + 1}")
    agent = str(step.get("agent") or "未知")
    title = str(step.get("title") or "")
    task = str(step.get("task") or "")
    acceptance = str(step.get("acceptance") or "")

    out = ""
    attempt = None
    tool_calls_count = None
    if isinstance(artifact, dict) and artifact:
        out = str(artifact.get("content") or "")
        attempt = artifact.get("attempt")
        tool_calls_count = artifact.get("tool_calls_count")
        # allow artifact to carry task/acceptance too
        task = task or str(artifact.get("task") or "")
        acceptance = acceptance or str(artifact.get("acceptance") or "")

    is_open = status in ("🔄", "⏳")  # current/in-progress
    header = f"{status} {step_id} ({agent})" + (f" — {title}" if title else "")
    if failure_count:
        header += f" | retry={failure_count}"
    if no_progress:
        header += " | no_progress=1"
    if tool_calls_count is not None:
        header += f" | tool_calls={tool_calls_count}"
    if attempt is not None:
        header += f" | attempt={attempt}"
    if history:
        header += f" | updates={len(history)}"

    out_full_len = len(out)
    out_disp = out
    truncated = False
    if max_output_chars and out_full_len > max_output_chars:
        out_disp = out[:max_output_chars]
        truncated = True

    fb_reason = ""
    fb_required = ""
    if isinstance(feedback, dict) and feedback:
        fb_reason = str(feedback.get("reason") or "")
        req = feedback.get("required_changes")
        fb_required = (
            "\n".join([str(x) for x in req])
            if isinstance(req, list)
            else str(req or "")
        )

    parts = []
    parts.append(f"<details {'open' if is_open else ''}>")
    parts.append(f"<summary>{_escape_md(header)}</summary>")
    parts.append("<div style='margin-top: 8px;'>")

    if task:
        parts.append(f"<div><b>任务</b><pre style='white-space:pre-wrap'>{_escape_md(task)}</pre></div>")
    if acceptance:
        parts.append(f"<div><b>验收标准</b><pre style='white-space:pre-wrap'>{_escape_md(acceptance)}</pre></div>")
    if fb_reason or fb_required:
        fb_block = ""
        if fb_reason:
            fb_block += f"原因：{fb_reason}\n"
        if fb_required:
            fb_block += f"要求：\n{fb_required}\n"
        parts.append(
            "<div><b>审阅/失败反馈</b>"
            f"<pre style='white-space:pre-wrap'>{_escape_md(fb_block)}</pre></div>"
        )

    # 简要展示同一步骤的多次更新（重试/工具返回/多轮）
    if history and len(history) > 1:
        tail = history[-8:]  # 只展示最近 8 条，避免 UI 过长
        lines = []
        for h in tail:
            a = (h.get("artifact") or {}) if isinstance(h, dict) else {}
            lines.append(
                "#%s node=%s tool_calls=%s attempt=%s" % (
                    h.get("trace_i"),
                    h.get("node"),
                    a.get("tool_calls_count"),
                    a.get("attempt"),
                )
            )
        parts.append(
            "<div><b>本步骤更新历史</b>"
            f"<pre style='white-space:pre-wrap'>{_escape_md(chr(10).join(lines))}</pre></div>"
        )

    if out_disp.strip():
        tip = ""
        if truncated:
            tip = f"（已截断：显示前 {max_output_chars} 字 / 共 {out_full_len} 字）"
        parts.append(
            "<div><b>产物输出</b> "
            f"<span style='color: #666;'>{_escape_md(tip)}</span>"
            f"<pre style='max-height: 280px; overflow:auto; white-space:pre-wrap'>{_escape_md(out_disp)}</pre></div>"
        )
    else:
        parts.append("<div><b>产物输出</b><div style='color:#666'>（暂无输出）</div></div>")

    parts.append("</div>")
    parts.append("</details>")
    return "\n".join(parts)


@st.cache_resource
def get_graph():
    # Initialize tools via MCP
    from llm import init_mcp_tools
    
    # We use asyncio.run to block until tools are loaded, as this is a sync cached function
    # Note: init_mcp_tools now uses temporary connections internally, so no manual cleanup needed here.
    tool_map = asyncio.run(init_mcp_tools())
    
    intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm = build_llm()
    return build_reflection_multi_agent_graph(
        intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm=responder_llm,
        tool_map=tool_map
    )


def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread-{uuid.uuid4()}"
    return st.session_state.thread_id


def render_steps_panel(session_state, traces_count, placeholders):
    """渲染右侧步骤面板 - 仅在有 traces 时显示详细信息"""

    if traces_count == 0:
        placeholders["progress"].info("暂无执行记录")
        placeholders["current_step"].info("等待用户输入...")
        placeholders["steps_detail"].info("等待开始执行...")
        return

    # 从 traces 中重建当前状态
    current_state = reconstruct_state_from_traces(session_state.traces)
    current_plan = current_state["plan"]
    current_step_idx = current_state["step_idx"]
    artifacts_by_step = current_state["artifacts"]
    artifacts_hist = current_state.get("artifact_history") or {}
    step_failures = current_state["step_failures"]
    no_progress = current_state["no_progress"]
    executed_steps = current_state.get("executed_steps") or []

    # 显示整体进度
    if current_plan and "steps" in current_plan:
        steps_norm = _normalize_steps(current_plan.get("steps"))
        total_steps = len(steps_norm)
        completed_steps = len(executed_steps) if executed_steps else len(artifacts_by_step)

        if total_steps > 0:
            progress = completed_steps / total_steps
            placeholders["progress"].progress(min(progress, 1.0))
            placeholders["progress"].caption(
                f"计划步骤：已完成 {completed_steps}/{total_steps} | 执行事件：{traces_count}"
            )

        # 显示当前正在执行的步骤
        if 0 <= current_step_idx < total_steps:
            current_step = steps_norm[current_step_idx]
            step_id = current_step.get("id", f"step{current_step_idx+1}")
            step_info = f"**当前步骤: {step_id}**\n\n**角色:** {current_step.get('agent', '未知')}\n\n**任务:** {current_step.get('task', '')}"

            if step_id in executed_steps:
                step_info += "\n\n✓ 本步骤已完成"
            else:
                failure_count = step_failures.get(step_id, 0)
                if failure_count > 0:
                    step_info += f"\n\n⚠️ 本步骤已重试 {failure_count} 次"
                elif no_progress.get(step_id, False):
                    step_info += "\n\n❌ 本步骤无进展"
                else:
                    step_info += "\n\n⏳ 本步骤正在执行中..."

            # 额外：显示该步骤的更新次数（同一 step 可能多轮/多次 tool_return）
            hist_n = len(artifacts_hist.get(step_id) or [])
            if hist_n:
                step_info += f"\n\n（本步骤已产生 {hist_n} 次更新）"

            placeholders["current_step"].markdown(step_info)
        else:
            placeholders["current_step"].info("所有步骤已完成")

        # 步骤详情显示（包含每个 step 的 task/acceptance/output/重试/工具统计）
        max_steps_show = 50
        detail_blocks = []
        last_feedback = current_state.get("last_feedback", {}) or {}
        for i, step in enumerate(steps_norm[:max_steps_show]):
            step_id = step.get("id", f"step{i+1}")
            if step_id in executed_steps:
                status = "✓"
            elif i == current_step_idx:
                status = "🔄"
            elif i < current_step_idx:
                status = "⏳"
            else:
                status = "⏳"

            artifact = artifacts_by_step.get(step_id, {})
            detail_blocks.append(
                _render_step_details_html(
                    step_idx=i,
                    step=step,
                    artifact=artifact,
                    history=artifacts_hist.get(step_id) or [],
                    status=status,
                    failure_count=int(step_failures.get(step_id, 0) or 0),
                    no_progress=bool(no_progress.get(step_id, False)),
                    feedback=last_feedback.get(step_id, {}) or {},
                )
            )

        if len(steps_norm) > max_steps_show:
            detail_blocks.append(
                f"<div style='color:#666'>（仅展示前 {max_steps_show} 个步骤；当前计划共 {len(steps_norm)} 个步骤）</div>"
            )

        placeholders["steps_detail"].markdown(
            "\n".join(detail_blocks), unsafe_allow_html=True
        )
    else:
        placeholders["progress"].info("等待计划生成...")
        placeholders["current_step"].info("等待计划生成...")
        placeholders["steps_detail"].info("等待计划生成...")


async def run_graph_stream(graph, user_text: str, thread_id: str):
    # 只传本轮增量输入，避免覆盖历史
    init_state = {"user_request": user_text}
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}

    async for upd in graph.astream(init_state, config=config, stream_mode="updates"):
        # upd: { node_name: patch }
        yield upd


def setup_logging(level=logging.INFO):
    print(f"Setting up logging with level {level}")
    # 【关键】在这里统一配置
    logging.basicConfig(
        level=level,
        format='time="%(asctime)s" level=%(levelname)s event=%(message)s',
        handlers=[
            logging.FileHandler("app.log"),  # 输出到文件
            # logging.StreamHandler(),  # 输出到控制台
        ],
        force=True,
    )


def main():
    workspace = utils.env("WORKSPACE.ROOT")
    if workspace is None or workspace == "":
        raise SystemExit("WORKSPACE.ROOT is not set")

    skills_dir = utils.env("SKILLS.DIR")
    if skills_dir is None or skills_dir == "":
        raise SystemExit("SKILLS.DIR is not set")

    intent_provider = utils.env("INTENT.PROVIDER")
    if intent_provider is None or intent_provider == "":
        raise SystemExit("INTENT.PROVIDER is not set")

    intent_model = utils.env("INTENT.MODEL")
    if intent_model is None or intent_model == "":
        raise SystemExit("INTENT.MODEL is not set")

    if not utils.check_llm_provider(intent_provider, intent_model):
        raise SystemExit(
            f"INTENT.PROVIDER={intent_provider} INTENT.MODEL={intent_model} is not configured"
        )

    planner_provider = utils.env("PLANNER.PROVIDER")
    if planner_provider is None or planner_provider == "":
        raise SystemExit("PLANNER.PROVIDER is not set")

    planner_model = utils.env("PLANNER.MODEL")
    if planner_model is None or planner_model == "":
        raise SystemExit("PLANNER.MODEL is not set")

    if not utils.check_llm_provider(planner_provider, planner_model):
        raise SystemExit(
            f"PLANNER.PROVIDER={planner_provider} PLANNER.MODEL={planner_model} is not configured"
        )

    agent_provider = utils.env("AGENT.PROVIDER")
    if agent_provider is None or agent_provider == "":
        raise SystemExit("AGENT.PROVIDER is not set")

    agent_model = utils.env("AGENT.MODEL")
    if agent_model is None or agent_model == "":
        raise SystemExit("AGENT.MODEL is not set")
    if not utils.check_llm_provider(agent_provider, agent_model):
        raise SystemExit(
            f"AGENT.PROVIDER={agent_provider} AGENT.MODEL={agent_model} is not configured"
        )

    search_backend = utils.env("SEARCH.BACKEND")
    if search_backend is None or search_backend == "":
        raise SystemExit("SEARCH.BACKEND is not set")

    # 设置日志
    log_level = utils.env("LOG.LEVEL", "INFO")
    if log_level.upper() == "DEBUG":
        setup_logging(logging.DEBUG)
    elif log_level.upper() == "INFO":
        setup_logging(logging.INFO)
    elif log_level.upper() == "WARNING":
        setup_logging(logging.WARNING)
    elif log_level.upper() == "ERROR":
        setup_logging(logging.ERROR)
    elif log_level.upper() == "CRITICAL":
        setup_logging(logging.CRITICAL)
    else:
        setup_logging(logging.INFO)

    logger = logging.getLogger(__name__)

    logger.info(
        f"[boot] WORKSPACE.ROOT={workspace} SKILLS.DIR={skills_dir} TOOL_REGISTRY={TOOL_REGISTRY} LOG_LEVEL={log_level}"
    )
    st.set_page_config(page_title="Agent Chat UI", layout="wide")

    # 左：聊天；右：步骤/日志
    col_chat, col_steps = st.columns([2, 1], gap="large")

    # 右侧进度面板的占位符字典
    placeholders = {}

    if "messages" not in st.session_state:
        st.session_state.messages = (
            []
        )  # [{"role": "user"/"assistant", "content": "..."}]
    if "traces" not in st.session_state:
        st.session_state.traces = []  # [{"node":..., "patch":...}, ...]

    graph = get_graph()
    thread_id = get_thread_id()

    with col_steps:
        st.subheader("任务监控")
        st.caption(f"thread_id = {thread_id}")

        # 使用 Tabs 分组显示
        tab_progress, tab_visual, tab_logs, tab_token = st.tabs(["执行进度", "时序图", "系统日志", "Token统计"])

        with tab_progress:
            # 创建占位符并存储到字典
            placeholders["progress"] = st.empty()
            placeholders["current_step"] = st.empty()
            placeholders["steps_detail"] = st.empty()
        
        with tab_visual:
            placeholders["mermaid"] = st.empty()
            
        with tab_logs:
            placeholders["logs"] = st.empty()
            
        with tab_token:
            placeholders["token"] = st.empty()

        if "ui_logs" not in st.session_state:
            st.session_state.ui_logs = []

        traces_count = len(st.session_state.traces)
        render_steps_panel(st.session_state, traces_count, placeholders)


        def update_token_display():
            if "total_token_usage" in st.session_state:
                usage = st.session_state.total_token_usage
                token_info = f"提示: {usage.get('prompt_tokens', 0)} | 完成: {usage.get('completion_tokens', 0)} | 总计: {usage.get('total_tokens', 0)}"
                placeholders["token"].markdown(token_info)
            else:
                placeholders["token"].text("暂无数据")

        update_token_display()
        
        # 初始渲染时序图
        if traces_count > 0:
            mermaid_code = generate_mermaid_sequence(st.session_state.traces)
            with placeholders["mermaid"]:
                render_mermaid(mermaid_code, height=500)
        else:
            placeholders["mermaid"].info("等待执行...")

    with col_chat:
        st.title("Chat")
        # 历史消息回放
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_text = st.chat_input("输入你的问题…")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)

            # assistant 流式区域
            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""

                async def _drive():
                    nonlocal acc
                    last_refresh = 0.0

                    try:
                        async for upd in run_graph_stream(graph, user_text, thread_id):
                            mem_snapshot("memory metric")
                            # 调试：记录更新结构
                            logger.debug(f"更新: {list(upd.keys())}")
                            for node, patch in upd.items():
                                st.session_state.traces.append(
                                    {"node": node, "patch": patch}
                                )
                                # 调试：记录节点和补丁键
                                if patch and isinstance(patch, dict):
                                    logger.debug(
                                        f"节点 {node} 补丁键: {list(patch.keys())}"
                                    )

                                # 提取并累计token使用量
                                if patch and isinstance(patch, dict):
                                    current_usage = None
                                    # 方法1：直接检查patch中是否有usage_metadata
                                    if "usage_metadata" in patch:
                                        usage = patch["usage_metadata"]
                                        if isinstance(usage, dict):
                                            current_usage = usage
                                    # 方法2：从messages中提取usage_metadata
                                    elif "messages" in patch:
                                        messages = patch["messages"]
                                        if messages:
                                            last_msg = messages[-1]
                                            # 从消息中提取usage_metadata
                                            usage = getattr(
                                                last_msg, "usage_metadata", None
                                            )
                                            if usage and isinstance(usage, dict):
                                                current_usage = usage

                                    if current_usage:
                                        # 初始化累计使用量
                                        if "total_token_usage" not in st.session_state:
                                            st.session_state.total_token_usage = {
                                                "prompt_tokens": 0,
                                                "completion_tokens": 0,
                                                "total_tokens": 0,
                                            }
                                        # 初始化历史记录
                                        if "token_history" not in st.session_state:
                                            st.session_state.token_history = []

                                        # 累计token使用量
                                        st.session_state.total_token_usage[
                                            "prompt_tokens"
                                        ] += current_usage.get("prompt_tokens", 0)
                                        st.session_state.total_token_usage[
                                            "completion_tokens"
                                        ] += current_usage.get("completion_tokens", 0)
                                        st.session_state.total_token_usage[
                                            "total_tokens"
                                        ] += current_usage.get("total_tokens", 0)

                                        # 添加到历史记录
                                        history_record = {
                                            "node": node,
                                            "prompt_tokens": current_usage.get(
                                                "prompt_tokens", 0
                                            ),
                                            "completion_tokens": current_usage.get(
                                                "completion_tokens", 0
                                            ),
                                            "total_tokens": current_usage.get(
                                                "total_tokens", 0
                                            ),
                                            "timestamp": time.time(),
                                        }
                                        st.session_state.token_history.append(
                                            history_record
                                        )

                                        logger.debug(
                                            f"累计token使用量: {st.session_state.total_token_usage}"
                                        )

                                        # 更新token显示占位符
                                        if placeholders and "token" in placeholders:
                                            usage = st.session_state.total_token_usage
                                            display_text = f"累计提示token: {usage.get('prompt_tokens', 0)}\n"
                                            display_text += f"累计完成token: {usage.get('completion_tokens', 0)}\n"
                                            display_text += f"累计总token: {usage.get('total_tokens', 0)}"

                                            # 显示token使用历史
                                            if (
                                                "token_history" in st.session_state
                                                and st.session_state.token_history
                                            ):
                                                display_text += "\n\n**Token使用历史:**"
                                                for i, record in enumerate(
                                                    st.session_state.token_history[-10:]
                                                ):  # 显示最近10条
                                                    display_text += f"\n{i+1}. 节点: {record.get('node', '未知')}, "
                                                    display_text += f"提示token: {record.get('prompt_tokens', 0)}, "
                                                    display_text += f"完成token: {record.get('completion_tokens', 0)}, "
                                                    display_text += f"总token: {record.get('total_tokens', 0)}"

                                            placeholders["token"].markdown(display_text)

                                # 检查是否需要触发UI更新
                                # ✅ 刷新右侧（建议节流，避免过于频繁）
                                now = time.time()
                                if now - last_refresh > 0.5:  # 放宽刷新频率
                                    render_steps_panel(
                                        st.session_state, len(st.session_state.traces), placeholders
                                    )
                                    
                                    # 刷新 Mermaid
                                    mermaid_code = generate_mermaid_sequence(st.session_state.traces)
                                    with placeholders["mermaid"]:
                                        render_mermaid(mermaid_code, height=500)
                                        
                                    last_refresh = now

                                # 注意：中间节点（researcher, solver, writer）的输出不再显示在左侧聊天区域
                                # 这些信息将在右侧进度面板中显示

                                # 如果是最终节点，把 final_answer 流式写出来
                                if node == "respond":
                                    final = (patch or {}).get("final_answer", "") or ""
                                    if final:
                                        acc = final
                                        placeholder.markdown(acc)

                                st.session_state.ui_logs.append(
                                    f"[{node}] keys={list(patch.keys()) if isinstance(patch, dict) else type(patch)}")
                                st.session_state.ui_logs = st.session_state.ui_logs[-300:]

                                # 额外：统计各节点出现次数，帮助对齐 app.log 的“执行轮次/节点更多”现象
                                try:
                                    counts = {}
                                    for t in st.session_state.traces[-2000:]:
                                        n = t.get("node")
                                        if not n:
                                            continue
                                        counts[n] = counts.get(n, 0) + 1
                                    counts_sorted = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                                    counts_text = ", ".join([f"{k}={v}" for k, v in counts_sorted[:25]])
                                    if len(counts_sorted) > 25:
                                        counts_text += ", ..."
                                except Exception:
                                    counts_text = ""

                                header = "节点计数（近 2000 条 trace 聚合）：\n" + (counts_text or "（暂无）")
                                body = "\n".join(st.session_state.ui_logs)
                                placeholders["logs"].code(header + "\n\n最近 updates：\n" + body, language="text")

                        return acc
                    finally:
                        # Ensure we cleanup any MCP sessions created in this loop
                        await CLIENT_MANAGER.close_all()

                final_answer = asyncio.run(_drive())

            render_steps_panel(st.session_state, len(st.session_state.traces), placeholders)

            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer}
            )


if __name__ == "__main__":
    main()

import asyncio
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
        "step_failures": {},
        "step_tool_stats": {},
        "no_progress": {},
        "last_feedback": {},
        "iter_count": 0,
        "done": False,
    }

    # 按时间顺序处理 traces（从旧到新），模拟状态更新
    for trace in traces:
        patch = trace.get("patch") or {}
        node = trace.get("node", "")

        # 调试：打印节点和补丁结构
        # print(f"节点: {node}, 补丁键: {list(patch.keys())}")

        # 应用补丁到当前状态
        # LangGraph 的补丁可能是嵌套的，例如 {"plan": {...}} 或 {"step_idx": 1}
        # 但也可能是更复杂的结构，例如 {"artifacts": {"step1": {...}}}
        for key, value in patch.items():
            if key in ["plan", "step_idx", "iter_count", "done"]:
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
                    else:
                        current_state[key].update(value)

    return current_state


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
    step_failures = current_state["step_failures"]
    no_progress = current_state["no_progress"]

    # 显示整体进度
    if current_plan and "steps" in current_plan:
        total_steps = len(current_plan["steps"])
        completed_steps = len(artifacts_by_step)

        if total_steps > 0:
            progress = completed_steps / total_steps
            placeholders["progress"].progress(min(progress, 1.0))
            placeholders["progress"].caption(
                f"已完成 {completed_steps}/{total_steps} 个步骤"
            )

        # 显示当前正在执行的步骤
        if 0 <= current_step_idx < total_steps:
            current_step = current_plan["steps"][current_step_idx]
            step_id = current_step.get("id", f"step{current_step_idx+1}")
            step_info = f"**当前步骤: {step_id}**\n\n**角色:** {current_step.get('agent', '未知')}\n\n**任务:** {current_step.get('task', '')}"

            if step_id in artifacts_by_step:
                step_info += "\n\n✓ 本步骤已完成"
            else:
                failure_count = step_failures.get(step_id, 0)
                if failure_count > 0:
                    step_info += f"\n\n⚠️ 本步骤已重试 {failure_count} 次"
                elif no_progress.get(step_id, False):
                    step_info += "\n\n❌ 本步骤无进展"
                else:
                    step_info += "\n\n⏳ 本步骤正在执行中..."

            placeholders["current_step"].markdown(step_info)
        else:
            placeholders["current_step"].info("所有步骤已完成")

        # 简化步骤详情显示
        steps_detail = []
        for i, step in enumerate(current_plan["steps"][:20]):  # 限制显示前20个步骤
            step_id = step.get("id", f"step{i+1}")
            step_agent = step.get("agent", "未知")

            if i < current_step_idx:
                status = "✓"
            elif i == current_step_idx:
                if step_id in artifacts_by_step:
                    status = "✓"
                else:
                    status = "🔄"
            else:
                status = "⏳"

            steps_detail.append(f"{status} {step_id} ({step_agent})")

        if len(current_plan["steps"]) > 20:
            steps_detail.append(f"... 共 {len(current_plan['steps'])} 个步骤")

        placeholders["steps_detail"].markdown("\n".join(steps_detail))
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
                                placeholders["logs"].code("\n".join(st.session_state.ui_logs), language="text")

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

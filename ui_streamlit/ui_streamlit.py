import html
import logging
import time
import traceback
import uuid
import streamlit as st
import streamlit.components.v1 as components
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.clawflow.utils import utils
from src.clawflow.config import env

# Your existing build functions (adjust imports for your project)
from src.clawflow.agents.llm import build_llm, build_reflection_multi_agent_graph
from my_tools import CLIENT_MANAGER

import os, gc, asyncio, tracemalloc
import psutil

from trace_visualizer import generate_mermaid_sequence

tracemalloc.start(25)
_proc = psutil.Process(os.getpid())

_last_top = 0.0
def mem_snapshot(tag: str, *, top: int = 0, interval_s: float = 60.0):
    rss = _proc.memory_info().rss / (1024 * 1024)
    vms = getattr(_proc.memory_info(), "vms", 0) / (1024 * 1024)

    cur_b, peak_b = tracemalloc.get_traced_memory()
    cur = cur_b / (1024 * 1024)
    peak = peak_b / (1024 * 1024)

    try:
        loop = asyncio.get_running_loop()
        tasks = len(asyncio.all_tasks(loop))
    except Exception:
        tasks = -1

    print(f"[MEM] {tag} rss={rss:.1f}MB vms={vms:.1f}MB py_cur={cur:.1f}MB py_peak={peak:.1f}MB gc={gc.get_count()} tasks={tasks}")

    global _last_top
    now = time.monotonic()
    if now - _last_top < interval_s:
        return
    _last_top = now
    # 可选：打印 tracemalloc top N 分配点（非常有用，但很吵）
    if top > 0:
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("lineno")
        print(f"[MEM] top {top} allocations:")
        for i, st in enumerate(stats[:top], 1):
            print(f"  {i:02d}. {st}")

_last_gc = 0.0
def maybe_gc(*, interval_s: float = 60.0):
    global _last_gc
    now = time.monotonic()
    if now - _last_gc >= interval_s:
        gc.collect()
        _last_gc = now

def mem_baseline(tag="baseline"):
    # 建议在系统稳定后做一次 baseline，然后 reset peak，后续峰值才有意义
    mem_snapshot(tag)
    tracemalloc.reset_peak()

_last = None
def mem_delta(tag: str):
    global _last
    rss = _proc.memory_info().rss
    cur, peak = tracemalloc.get_traced_memory()
    if _last is None:
        _last = (rss, cur)
        print(f"[MEMΔ] {tag} init")
        return
    drss = (rss - _last[0]) / (1024 * 1024)
    dcur = (cur - _last[1]) / (1024 * 1024)
    _last = (rss, cur)
    print(f"[MEMΔ] {tag} Δrss={drss:+.1f}MB Δpy_cur={dcur:+.1f}MB")

MAX_TRACES = int(os.getenv("MAX_TRACES", "100"))
def append_trace(traces: list, item):
    traces.append(item)
    if len(traces) > MAX_TRACES:
        del traces[: len(traces) - MAX_TRACES]

def render_mermaid(code: str, height=600):
    """
    Render Mermaid diagrams via HTML/JS.
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
    """Rebuild current state from traces."""
    current_state = {
        "plan": None,
        "step_idx": 0,
        "artifacts": {},
        # Track multiple outputs for the same step (retries/tool returns/multi-turn)
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

    # Process traces in chronological order (old -> new) to simulate state updates
    for trace_i, trace in enumerate(traces):
        patch = trace.get("patch") or {}
        node = trace.get("node", "")

        # Debug: print node and patch structure
        # print(f"node: {node}, patch_keys: {list(patch.keys())}")

        # Apply patch to current state
        # LangGraph patches can be nested, e.g. {"plan": {...}} or {"step_idx": 1}
        # They can also be more complex, e.g. {"artifacts": {"step1": {...}}}
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
                    # If it's a dict, merge updates
                    if key == "artifacts":
                        # artifacts may contain outputs for multiple steps; merge them
                        for step_id, artifact in value.items():
                            current_state[key][step_id] = artifact
                            # Accumulate history for UI (it may look like many rounds)
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
    agent = str(step.get("../agent") or "未知")
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

    # Briefly show multiple updates for the same step (retries/tool returns/multi-turn)
    if history and len(history) > 1:
        tail = history[-8:]  # Show only the last 8 entries to avoid an overly long UI
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
    """
    缓存“重资产”：tool_map + llms + cp_path
    """
    cp_path = utils.getenv("CHECKPOINTER.DB_PATH", "./checkpoints.sqlite")

    # Initialize tools via MCP (sync cached function -> block with asyncio.run)
    from src.clawflow.agents.llm import init_mcp_tools
    tool_map = asyncio.run(init_mcp_tools())

    intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm = build_llm()

    return {
        "cp_path": cp_path,
        "tool_map": tool_map,
        "llms": (intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm),
    }



def get_thread_id():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread-{uuid.uuid4()}"
    return st.session_state.thread_id


def render_steps_panel(session_state, traces_count, placeholders):
    """Render the right-side steps panel (only show details when traces exist)."""

    if traces_count == 0:
        placeholders["progress"].info("暂无执行记录")
        placeholders["current_step"].info("等待用户输入...")
        placeholders["steps_detail"].info("等待开始执行...")
        return

    # Rebuild current state from traces
    current_state = reconstruct_state_from_traces(session_state.traces)
    current_plan = current_state["plan"]
    current_step_idx = current_state["step_idx"]
    artifacts_by_step = current_state["artifacts"]
    artifacts_hist = current_state.get("artifact_history") or {}
    step_failures = current_state["step_failures"]
    no_progress = current_state["no_progress"]
    executed_steps = current_state.get("executed_steps") or []

    # Show overall progress
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

        # Show the currently running step
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

            # Extra: show update count for this step (may have multiple rounds/tool returns)
            hist_n = len(artifacts_hist.get(step_id) or [])
            if hist_n:
                step_info += f"\n\n（本步骤已产生 {hist_n} 次更新）"

            placeholders["current_step"].markdown(step_info)
        else:
            placeholders["current_step"].info("所有步骤已完成")

        # Step details (task/acceptance/output/retries/tool stats)
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


async def run_graph_stream(assets, user_text: str, thread_id: str):
    init_state = {"user_request": user_text}
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}

    intent_llm, planner_llm, agent_llm, reflector_llm, responder_llm = assets["llms"]
    tool_map = assets["tool_map"]
    cp_path = assets["cp_path"]

    async with AsyncSqliteSaver.from_conn_string(cp_path) as cp:
        graph = build_reflection_multi_agent_graph(
            intent_llm, planner_llm, agent_llm, reflector_llm,
            responder_llm=responder_llm,
            tool_map=tool_map,
            checkpointer=cp,
        )

        async for upd in graph.astream(init_state, config=config, stream_mode="updates"):
            yield upd


def setup_logging(level=logging.INFO):
    print(f"Setting up logging with level {level}")
    # Key: centralize logging config here
    logging.basicConfig(
        level=level,
        format='time="%(asctime)s" level=%(levelname)s event=%(message)s',
        handlers=[
            logging.FileHandler("./app.log"),  # Write to file
            # logging.StreamHandler(),  # Write to console
        ],
        force=True,
    )


def main():
    workspace = env("WORKSPACE.ROOT")
    if workspace is None or workspace == "":
        raise SystemExit("WORKSPACE.ROOT is not set")
    os.environ["WORKSPACE_ROOT"] = workspace
    skills_dir = env("SKILLS.DIR")
    if skills_dir is None or skills_dir == "":
        raise SystemExit("SKILLS.DIR is not set")

    intent_provider = env("INTENT.PROVIDER")
    if intent_provider is None or intent_provider == "":
        raise SystemExit("INTENT.PROVIDER is not set")
    intent_model = env("INTENT.MODEL")
    if intent_model is None or intent_model == "":
        raise SystemExit("INTENT.MODEL is not set")

    if not utils.check_llm_provider(intent_provider, intent_model):
        raise SystemExit(
            f"INTENT.PROVIDER={intent_provider} INTENT.MODEL={intent_model} is not configured"
        )
    planner_provider = env("PLANNER.PROVIDER")
    if planner_provider is None or planner_provider == "":
        raise SystemExit("PLANNER.PROVIDER is not set")

    planner_model = env("PLANNER.MODEL")
    if planner_model is None or planner_model == "":
        raise SystemExit("PLANNER.MODEL is not set")

    if not utils.check_llm_provider(planner_provider, planner_model):
        raise SystemExit(
            f"PLANNER.PROVIDER={planner_provider} PLANNER.MODEL={planner_model} is not configured"
        )

    agent_provider = env("AGENT.PROVIDER")
    if agent_provider is None or agent_provider == "":
        raise SystemExit("AGENT.PROVIDER is not set")

    agent_model = env("AGENT.MODEL")
    if agent_model is None or agent_model == "":
        raise SystemExit("AGENT.MODEL is not set")
    if not utils.check_llm_provider(agent_provider, agent_model):
        raise SystemExit(
            f"AGENT.PROVIDER={agent_provider} AGENT.MODEL={agent_model} is not configured"
        )

    reflector_provider = env("REFLECTOR.PROVIDER")
    if reflector_provider is None or reflector_provider == "":
        raise SystemExit("REFLECTOR.PROVIDER is not set")

    reflector_model = env("REFLECTOR.MODEL")
    if reflector_model is None or reflector_model == "":
        raise SystemExit("REFLECTOR.MODEL is not set")

    if not utils.check_llm_provider(reflector_provider, reflector_model):
        raise SystemExit(
            f"REFLECTOR.PROVIDER={reflector_provider} REFLECTOR.MODEL={reflector_model} is not configured"
        )

    search_backend = env("SEARCH.BACKEND")
    if search_backend is None or search_backend == "":
        raise SystemExit("SEARCH.BACKEND is not set")
    os.environ["SEARCH_BACKEND"] = search_backend

    searxng_url = env("SEARCH.URL")
    if searxng_url:
        os.environ["SEARXNG_URL"] = searxng_url

    tavily_api_key = env("TAVIY.API_KEY")
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    serpapi_key = env("SERPAPI.API_KEY")
    if serpapi_key:
        os.environ["SERPAPI_API_KEY"] = serpapi_key

    bing_api_key = env("BING.API_KEY")
    if bing_api_key:
        os.environ["BINGAPI_API_KEY"] = bing_api_key

    # Configure logging
    log_level = env("LOG.LEVEL", "INFO")
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
        f"[boot] WORKSPACE.ROOT={workspace} SKILLS.DIR={skills_dir} LOG_LEVEL={log_level}"
    )
    st.set_page_config(page_title="Agent Chat UI", layout="wide")

    # Left: chat; Right: steps/logs
    col_chat, col_steps = st.columns([2, 1], gap="large")

    # Placeholder dict for the right-side progress panel
    placeholders = {}

    if "messages" not in st.session_state:
        st.session_state.messages = (
            []
        )  # [{"role": "user"/"assistant", "content": "..."}]
    if "traces" not in st.session_state:
        st.session_state.traces = []  # [{"node":..., "patch":...}, ...]

    assets = get_graph()
    thread_id = get_thread_id()
    with col_steps:
        st.subheader("任务监控")
        st.caption(f"thread_id = {thread_id}")

        # Group display using Tabs
        tab_progress, tab_visual, tab_logs, tab_token = st.tabs(["执行进度", "时序图", "系统日志", "Token统计"])

        with tab_progress:
            # Create placeholders and store them in the dict
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
        
        # Initial render of the sequence diagram
        if traces_count > 0:
            mermaid_code = generate_mermaid_sequence(st.session_state.traces)
            with placeholders["mermaid"]:
                render_mermaid(mermaid_code, height=500)
        else:
            placeholders["mermaid"].info("等待执行...")

    with col_chat:
        st.title("Chat")
        # Replay chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_text = st.chat_input("输入你的问题…")
        if user_text:
            st.session_state.traces = []
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)

            # assistant streaming area
            with st.chat_message("assistant"):
                placeholder = st.empty()
                acc = ""

                async def _drive():
                    nonlocal acc
                    last_refresh = 0.0

                    try:
                        if os.getenv("MEMORY_METRICS_AND_GC", "0") == "1":
                            mem_baseline()
                        async for upd in run_graph_stream(assets, user_text, thread_id):
                            if os.getenv("MEMORY_METRICS_AND_GC", "0") == "1":
                                mem_snapshot("before gc", top=10)
                                maybe_gc()
                                mem_snapshot("after gc", top=10)
                            # Debug: record update structure
                            logger.debug(f"更新: {list(upd.keys())}")
                            for node, patch in upd.items():
                                append_trace(st.session_state.traces,
                                    {"node": node, "patch": patch}
                                )
                                # Debug: record node and patch keys
                                if patch and isinstance(patch, dict):
                                    logger.debug(
                                        f"节点 {node} 补丁键: {list(patch.keys())}"
                                    )

                                # Extract and accumulate token usage
                                if patch and isinstance(patch, dict):
                                    current_usage = None
                                    # Method 1: check usage_metadata directly in patch
                                    if "usage_metadata" in patch:
                                        usage = patch["usage_metadata"]
                                        if isinstance(usage, dict):
                                            current_usage = usage
                                    # Method 2: extract usage_metadata from messages
                                    elif "messages" in patch:
                                        messages = patch["messages"]
                                        if messages:
                                            last_msg = messages[-1]
                                            # Extract usage_metadata from the message
                                            usage = getattr(
                                                last_msg, "usage_metadata", None
                                            )
                                            if usage and isinstance(usage, dict):
                                                current_usage = usage

                                    if current_usage:
                                        # Initialize cumulative usage
                                        if "total_token_usage" not in st.session_state:
                                            st.session_state.total_token_usage = {
                                                "prompt_tokens": 0,
                                                "completion_tokens": 0,
                                                "total_tokens": 0,
                                            }
                                        # Initialize history
                                        if "token_history" not in st.session_state:
                                            st.session_state.token_history = []

                                        # Accumulate token usage
                                        st.session_state.total_token_usage[
                                            "prompt_tokens"
                                        ] += current_usage.get("prompt_tokens", 0)
                                        st.session_state.total_token_usage[
                                            "completion_tokens"
                                        ] += current_usage.get("completion_tokens", 0)
                                        st.session_state.total_token_usage[
                                            "total_tokens"
                                        ] += current_usage.get("total_tokens", 0)

                                        # Append to history
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

                                        # Update token display placeholder
                                        if placeholders and "token" in placeholders:
                                            usage = st.session_state.total_token_usage
                                            display_text = f"累计提示token: {usage.get('prompt_tokens', 0)}\n"
                                            display_text += f"累计完成token: {usage.get('completion_tokens', 0)}\n"
                                            display_text += f"累计总token: {usage.get('total_tokens', 0)}"

                                            # Show token usage history
                                            if (
                                                "token_history" in st.session_state
                                                and st.session_state.token_history
                                            ):
                                                display_text += "\n\n**Token使用历史:**"
                                                for i, record in enumerate(
                                                    st.session_state.token_history[-10:]
                                                ):  # Show last 10 entries
                                                    display_text += f"\n{i+1}. 节点: {record.get('node', '未知')}, "
                                                    display_text += f"提示token: {record.get('prompt_tokens', 0)}, "
                                                    display_text += f"完成token: {record.get('completion_tokens', 0)}, "
                                                    display_text += f"总token: {record.get('total_tokens', 0)}"

                                            placeholders["token"].markdown(display_text)

                                # Check whether UI refresh is needed
                                # ✅ Refresh right panel (throttle to avoid being too frequent)
                                now = time.time()
                                if now - last_refresh > 0.5:  # Loosen refresh frequency
                                    render_steps_panel(
                                        st.session_state, len(st.session_state.traces), placeholders
                                    )
                                    
                                    # Refresh Mermaid
                                    mermaid_code = generate_mermaid_sequence(st.session_state.traces)
                                    with placeholders["mermaid"]:
                                        render_mermaid(mermaid_code, height=500)
                                        
                                    last_refresh = now

                                # Note: outputs from intermediate nodes (researcher/solver/writer) are no longer shown in the left chat area
                                # These will be shown in the right progress panel

                                # If this is the final node, stream final_answer
                                if node == "respond":
                                    final = (patch or {}).get("final_answer", "") or ""
                                    if final:
                                        acc = final
                                        placeholder.markdown(acc)

                                st.session_state.ui_logs.append(
                                    f"[{node}] keys={list(patch.keys()) if isinstance(patch, dict) else type(patch)}")
                                st.session_state.ui_logs = st.session_state.ui_logs[-300:]

                                # Extra: count node occurrences to help reconcile app.log (more rounds/nodes)
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
    try:
        main()
    except Exception:
        traceback.print_exc()

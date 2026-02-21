import hashlib
import json
import logging
import time
from typing import Literal, Dict, Any, List, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from src.clawflow.config import env
from src.clawflow.utils import utils, retry_with_backoff, msg_preview, to_dumpable, extract_resp_id, \
    extract_finish_reason, extract_tool_calls, truncate, ai_meta

logger = logging.getLogger(__name__)


class AgentStepModel(BaseModel):
    id: str
    title: str
    agent: Literal["researcher", "solver", "writer", "code_researcher", "code_reviewer"]
    task: str
    acceptance: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)


class PlanModel(BaseModel):
    version: int = 1
    objective: str = ""
    steps: List[AgentStepModel] = Field(default_factory=list)


class ReflectModel(BaseModel):
    decision: Literal["accept", "retry", "revise_plan", "finish", "generate_next_step"]
    reason: str
    required_changes: List[str] = Field(default_factory=list)
    # Avoid Optional -> anyOf/null (more stable for local structured output)
    plan_patch: PlanModel = Field(default_factory=PlanModel)
    next_step: Optional[AgentStepModel] = None  # Used to dynamically generate the next step

class PlanReviewModel(BaseModel):
    decision: Literal["approve", "reject"]
    feedback: str = Field(description="If rejected, provide specific feedback on how to improve the plan.")

class AgentStep(TypedDict, total=False):
    id: str
    title: str
    agent: Literal["researcher", "solver", "writer", "code_researcher", "code_reviewer"]
    task: str
    acceptance: str
    inputs: Dict[str, Any]
    depends_on: List[str]


class Plan(TypedDict, total=False):
    version: int
    objective: str
    steps: List[AgentStep]


class AgentState(TypedDict, total=False):
    # ---- conversation & planning ----
    user_request: str
    intent: Dict[str, Any]

    plan: Dict[str, Any]
    step_idx: int
    done: bool
    next_node: str
    last_agent_role: str

    # ---- execution memory (global, for audit) ----
    # Keep full messages for audit/debugging, but don't feed them to the LLM every time
    messages: Annotated[List[AnyMessage], add_messages]

    # ---- step-local message cursor ----
    # Record messages length at the start of each step.
    # agent_node only takes messages[start_idx:] as the step-local context
    step_msg_start: Dict[str, int]

    # ---- artifacts & bookkeeping ----
    artifacts: Dict[str, Dict[str, Any]]      # step_id -> {content, tool_calls_count, ...}
    executed_steps: List[str]
    seen_step_hashes: List[str]

    step_failures: Dict[str, int]
    last_feedback: Dict[str, Any]             # step_id -> {reason, required_changes, ...}
    last_output_hash: Dict[str, str]
    no_progress: Dict[str, bool]
    step_tool_stats: Dict[str, int]

    iter_count: int
    max_iters: int

    # ---- replanning control ----
    need_plan_update: bool
    dynamic_plan_feedback: str
    plan_review_feedback: Optional[str]

    # ---- logs / metrics ----
    usage_metadata: Dict[str, Any]

    final_answer: str

# --- Intent Analysis ---
class IntentEntity(BaseModel):
    type: Literal[
        "date",
        "time_range",
        "location",
        "person",
        "org",
        "product",
        "file",
        "url",
        "codebase",
        "concept",
        "other",
    ] = Field(..., description="实体类型")
    value: str = Field(..., description="实体值（原文或规范化值）")


class IntentModel(BaseModel):
    """Structured intent analysis model."""

    # 1) What this request is asking to do
    task_type: Literal[
        "research",  # Look up/verify facts/needs citations
        "analysis",  # Analysis (data analysis/comparison/inference)
        "planning",  # Generate a plan / break down steps
        "writing",  # Writing/polishing/summarization
        "coding",  # Implement/modify code
        "debugging",  # Debug/troubleshoot
        "translation",  # Translation
        "summarization",  # Summarization
        "qa",  # General Q&A
        "other",
    ] = Field(
        ...,
        description="任务类型（通用分类）。参考任务类型详解：research, analysis, planning, writing, coding, debugging, translation, summarization, qa, other。",
    )

    # 2) User goal (short, planner-friendly)
    user_goal: str = Field(
        ...,
        description="用户目标/期望产出（尽量一句话）。应简洁明确，便于规划器直接理解。",
    )

    # 3) Domains (optional; don't force classification)
    domains: List[str] = Field(
        default_factory=list,
        description="可选：领域标签列表，如 finance/software/legal/health/education/...；不确定则空列表 []。用于指导工具选择和知识范围。",
    )

    # 4) Deliverable type (planner uses it to choose a template)
    deliverable: Literal[
        "direct_answer",  # Direct answer
        "step_plan",  # Executable steps
        "report",  # Structured report
        "code_patch",  # Patch/code snippet
        "table",  # Table/checklist
        "json",  # JSON output
        "other",
    ] = Field(
        default="direct_answer", description="期望交付物形式。指导规划器选择输出模板。"
    )

    # 5) Key slots (no business binding; extract only explicit entities)
    entities: List[IntentEntity] = Field(
        default_factory=list,
        description="从用户原文抽取的关键实体。仅抽取明确提及的实体，必要时规范化（如日期转 YYYY-MM-DD）。",
    )

    # 6) Tool/web preference (only need/suggestion; no actual calls in intent stage)
    need_web: bool = Field(
        default=False,
        description="是否需要联网检索/获取最新信息。当任务需要外部实时数据时设为 true。",
    )
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="建议使用的工具能力名称（如 web_search/file_search/calculator/db_query/...）。根据任务类型推荐，但不强制使用。",
    )

    # 7) Constraints and preferences
    constraints: List[str] = Field(
        default_factory=list,
        description="用户约束/偏好（格式、风格、范围、语言等）。例如：'输出为表格'、'使用中文'、'不超过500字'。",
    )

    # 8) Missing info (don't ask here; mark for planner to decide)
    missing_info: List[str] = Field(
        default_factory=list,
        description="完成任务可能缺失的信息点（例如：标的、时间范围、数据源等）。供规划器决定是否向用户追问。",
    )

    # 9) Output language (can follow system default)
    output_language: Literal["zh", "en"] = Field(
        default="zh", description="输出语言。zh 表示中文，en 表示英文。默认为中文。"
    )

    # 10) Role presets (define responsibilities, scope, default behavior per role)
    role_preset: Dict[str, str] = Field(
        default_factory=dict,
        description="为每个可能参与的角色（researcher/solver/writer）定义预设职责和能力范围。例如：{'researcher': '擅长信息搜集、事实核实、外部数据获取', 'solver': '擅长逻辑推理、数据分析、算法实现', 'writer': '擅长文本整理、报告撰写、语言润色'}。预设信息将帮助规划器更准确地分配任务，发挥每个角色的专长。",
    )


def hash_step(step: AgentStep) -> str:
    stable = {
        "agent": step.get("", ""),
        "title": step.get("title", ""),
        "task": step.get("task", ""),
        "acceptance": step.get("acceptance", ""),
        "inputs": step.get("inputs", {}),
        "depends_on": step.get("depends_on", []),
    }
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def current_step(state: AgentState) -> AgentStep:
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    idx = state.get("step_idx", 0)
    if not isinstance(steps, list):
        return {}
    if not (0 <= idx < len(steps)):
        return {}

    st = steps[idx]
    # Pydantic model
    if hasattr(st, "model_dump"):
        try:
            st = st.model_dump()
        except Exception:
            pass
    # If step is not a dict (e.g., str), avoid crashing downstream.
    return st if isinstance(st, dict) else {}

@utils.timer
async def async_invoke_chat_with_retry(
    llm, msgs, *, role="unknown", retries=2, base_sleep=0.5
) -> AIMessage:
    # Prevent scope pollution if def log_event / log_event=... appears inside this function
    _log_event = globals().get("log_event")
    if not callable(_log_event):
        import logging

        def _log_event(event: str, **kwargs):
            logging.getLogger("agent_flow").info("event=%s %s", event, kwargs)

    _OllamaResponseError = globals().get("OllamaResponseError")
    if not isinstance(_OllamaResponseError, type) or not issubclass(
        _OllamaResponseError, BaseException
    ):
        _OllamaResponseError = Exception

    @utils.timer
    @retry_with_backoff(retries=retries, base_sleep=base_sleep)
    async def _invoke_once() -> AIMessage:
        _log_event("llm.chat.call", try_i=0, role=role, msgs=msg_preview(msgs))

        def _has_orphan_tool_message(msgs: list[BaseMessage]) -> bool:
            seen_tool_calls = False
            for m in msgs:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    seen_tool_calls = True
                if isinstance(m, ToolMessage) and not seen_tool_calls:
                    return True
            return False

        if _has_orphan_tool_message(msgs):
            logger.warning(f"orphan tool message {msg_preview(msgs, n=6)}")

        # Automatically inject time context
        msgs_with_time = utils.inject_time_context(msgs)
        if msgs_with_time != msgs:
            logger.debug(
                f"Time context injected for {role}, original_len={len(msgs)}, new_len={len(msgs_with_time)}"
            )

        msgs_with_time = utils.filter_messages_for_llm(msgs_with_time)
        start = time.perf_counter()

        resp = await llm.ainvoke(msgs_with_time)
        end = time.perf_counter()
        logger.info(f"{__name__} 耗时: {end - start:.6f} 秒")

        # Always log minimal meta; never log full model content (avoid leaking "thinking"/raw output)
        resp_id = extract_resp_id(resp)
        finish_reason = extract_finish_reason(resp)
        tool_calls = extract_tool_calls(resp)
        _log_event(
            "llm.chat.meta",
            role=role,
            resp_id=resp_id,
            finish_reason=finish_reason,
            tool_calls_cnt=len(tool_calls),
        )

        if isinstance(resp, AIMessage):
            _log_event("llm.chat.ok", try_i=0, role=role, llm_meta=ai_meta(resp))
            return resp

        msg = AIMessage(content=str(getattr(resp, "content", resp)))
        _log_event("llm.chat.ok", try_i=0, role=role, llm_meta=ai_meta(msg))
        return msg

    try:
        return await _invoke_once()
    except Exception as e:
        # Final exception after all retries are exhausted
        _log_event("llm.chat.err", try_i=0, role=role, err=f"{type(e).__name__}: {e}")

        # Generate a user-friendly error message
        error_type = type(e).__name__
        error_msg = str(e)

        # Provide different friendly messages based on error type
        friendly_message = None

        # Check whether it's a network connection error
        if error_type in ("ConnectError", "ConnectionError", "gaierror"):
            if "nodename nor servname provided" in error_msg or "DNS" in error_msg:
                friendly_message = "无法连接到AI服务，请检查网络连接和DNS设置。"
            else:
                friendly_message = "网络连接出现问题，请稍后重试。"
        elif error_type in ("TimeoutError", "TimeoutException", "socket.timeout"):
            friendly_message = "请求超时，请检查网络状况或稍后重试。"
        elif (
            "rate limit" in error_msg.lower()
            or "too many requests" in error_msg.lower()
        ):
            friendly_message = "请求过于频繁，请稍后重试。"
        elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            friendly_message = "API认证失败，请检查API密钥配置。"

        # If error type can't be recognized, use a generic message
        if not friendly_message:
            friendly_message = "AI服务暂时不可用，请稍后重试。"

        # Return a friendly message while logging detailed error
        return AIMessage(
            content=friendly_message
            + f"\n\n（技术细节：{error_type}: {error_msg[:200]}）"
        )

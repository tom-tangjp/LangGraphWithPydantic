import hashlib
import json
import logging
import time
from typing import Literal, Dict, Any, List, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from utils import (
    msg_preview,
    truncate,
    to_dumpable,
    extract_resp_id,
    extract_finish_reason,
    extract_tool_calls,
    retry_with_backoff,
    ai_meta,
    env,
)
import utils

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
    # 避免 Optional -> anyOf/null（本地 structured 输出更稳）
    plan_patch: PlanModel = Field(default_factory=PlanModel)
    next_step: Optional[AgentStepModel] = None  # 用于动态生成下一步

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
    # 仍然保留全量 messages 作为审计与调试，但不再每次都喂给 LLM
    messages: Annotated[List[AnyMessage], add_messages]

    # ---- step-local message cursor ----
    # 记录每个 step 开始时 messages 的长度。
    # agent_node 只取 messages[start_idx:] 作为“本 step 局部上下文”
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
    traces: List[Dict[str, Any]]
    usage_metadata: Dict[str, Any]

    final_answer: str

### 意图分析
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

    # 1) 这次请求“要做什么”
    task_type: Literal[
        "research",  # 查资料/核验事实/需要引用
        "analysis",  # 分析（含数据分析、对比、推断）
        "planning",  # 生成计划/拆解步骤
        "writing",  # 写作/润色/总结
        "coding",  # 写代码/实现
        "debugging",  # 定位问题/修 bug/排障
        "translation",  # 翻译
        "summarization",  # 摘要/提炼
        "qa",  # 一般问答
        "other",
    ] = Field(
        ...,
        description="任务类型（通用分类）。参考任务类型详解：research, analysis, planning, writing, coding, debugging, translation, summarization, qa, other。",
    )

    # 2) 用户目标（短句，便于 planner 直接消费）
    user_goal: str = Field(
        ...,
        description="用户目标/期望产出（尽量一句话）。应简洁明确，便于规划器直接理解。",
    )

    # 3) 领域（可空，不强行归类）
    domains: List[str] = Field(
        default_factory=list,
        description="可选：领域标签列表，如 finance/software/legal/health/education/...；不确定则空列表 []。用于指导工具选择和知识范围。",
    )

    # 4) 交付物形式（planner 可据此选模板）
    deliverable: Literal[
        "direct_answer",  # 直接回答
        "step_plan",  # 可执行步骤
        "report",  # 结构化报告
        "code_patch",  # 补丁/代码片段
        "table",  # 表格/清单
        "json",  # JSON 输出
        "other",
    ] = Field(
        default="direct_answer", description="期望交付物形式。指导规划器选择输出模板。"
    )

    # 5) 关键信息槽位（不做业务绑定，仅抽取显式实体）
    entities: List[IntentEntity] = Field(
        default_factory=list,
        description="从用户原文抽取的关键实体。仅抽取明确提及的实体，必要时规范化（如日期转 YYYY-MM-DD）。",
    )

    # 6) 工具/联网倾向（只做“需要与否/建议”，不在 intent 阶段真的调用）
    need_web: bool = Field(
        default=False,
        description="是否需要联网检索/获取最新信息。当任务需要外部实时数据时设为 true。",
    )
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="建议使用的工具能力名称（如 web_search/file_search/calculator/db_query/...）。根据任务类型推荐，但不强制使用。",
    )

    # 7) 约束与偏好
    constraints: List[str] = Field(
        default_factory=list,
        description="用户约束/偏好（格式、风格、范围、语言等）。例如：'输出为表格'、'使用中文'、'不超过500字'。",
    )

    # 8) 缺失信息（不在这里追问，只标注给 planner 决定是否追问）
    missing_info: List[str] = Field(
        default_factory=list,
        description="完成任务可能缺失的信息点（例如：标的、时间范围、数据源等）。供规划器决定是否向用户追问。",
    )

    # 9) 输出语言（可按你的系统默认）
    output_language: Literal["zh", "en"] = Field(
        default="zh", description="输出语言。zh 表示中文，en 表示英文。默认为中文。"
    )

    # 10) 角色预设（为每个可能的执行角色定义职责、能力范围和默认行为模式）
    role_preset: Dict[str, str] = Field(
        default_factory=dict,
        description="为每个可能参与的角色（researcher/solver/writer）定义预设职责和能力范围。例如：{'researcher': '擅长信息搜集、事实核实、外部数据获取', 'solver': '擅长逻辑推理、数据分析、算法实现', 'writer': '擅长文本整理、报告撰写、语言润色'}。预设信息将帮助规划器更准确地分配任务，发挥每个角色的专长。",
    )


def hash_step(step: AgentStep) -> str:
    stable = {
        "agent": step.get("agent", ""),
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
    if 0 <= idx < len(steps):
        return steps[idx]
    return {}

@utils.timer
async def async_invoke_chat_with_retry(
    llm, msgs, *, role="unknown", retries=2, base_sleep=0.5
) -> AIMessage:
    # 防止函数体内（哪怕是后面）出现 def log_event / log_event=... 导致作用域污染
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

        # 自动注入时间上下文
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

        # 建议：把这段打印放到开关里，否则会非常吵
        if env("AGENT_LOG_LLM_DUMP", "0") == "1":
            try:
                msgs_data = [to_dumpable(m) for m in msgs]
                print(f"--------- {role} Question: ---------")
                print(json.dumps(msgs_data, ensure_ascii=False, indent=2, default=str))

                print(f"--------- {role} Answer 耗时:{end - start:.6f} 秒: ---------")
                print(json.dumps(resp, ensure_ascii=False, indent=2, default=str))
            except Exception:
                pass

        if env("LOG.LLM.CONTENT", "0") == "1":
            resp_id = extract_resp_id(resp)
            finish_reason = extract_finish_reason(resp)
            tool_calls = extract_tool_calls(resp)
            _log_event(
                "llm.chat.meta",
                role=role,
                resp_id=resp_id,
                finish_reason=finish_reason,
                tool_calls=truncate(json.dumps(tool_calls, ensure_ascii=False), 2000),
            )
            _log_event(
                "llm.chat.content",
                try_i=0,
                role=role,
                preview=truncate(getattr(resp, "content", ""), 800),
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
        # 所有重试耗尽后的最终异常
        _log_event("llm.chat.err", try_i=0, role=role, err=f"{type(e).__name__}: {e}")

        # 生成用户友好的错误消息
        error_type = type(e).__name__
        error_msg = str(e)

        # 根据错误类型提供不同的友好消息
        friendly_message = None

        # 检查是否为网络连接错误
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

        # 如果无法识别错误类型，使用通用消息
        if not friendly_message:
            friendly_message = "AI服务暂时不可用，请稍后重试。"

        # 返回友好的错误消息，同时记录详细错误
        return AIMessage(
            content=friendly_message
            + f"\n\n（技术细节：{error_type}: {error_msg[:200]}）"
        )

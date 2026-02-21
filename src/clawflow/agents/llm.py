import json
import re
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal, get_args, Optional, Dict, List

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    ToolMessage,
    HumanMessage,
    BaseMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Gemini (Google AI Studio) support (optional dependency)
_GEMINI_IMPORT_ERROR: Exception | None = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as e:
    # NOTE: do not crash the whole app if Gemini deps are missing/misconfigured.
    # Keep the error for later diagnostics.
    ChatGoogleGenerativeAI = None
    _GEMINI_IMPORT_ERROR = e
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

from src.clawflow import utils
from src.clawflow.agents import internal_prompt
from src.clawflow.agents.internal_prompt import INTENT_SYSTEM, PLAN_SYSTEM, PLAN_REVIEW_SYSTEM, AGENT_SYSTEMS, \
    TIME_FALLBACK_CONTENT, AGENT_RETRY_INSTRUCTION, RESPOND_SYSTEM, REFLECT_SYSTEM
from src.clawflow.agents.tool_cheatsheet import render_tool_cheatsheet
from src.clawflow.config import env

try:
    from pydantic_ai.models.openai import OpenAIChatModel
except ImportError:
    from pydantic_ai.models.openai import OpenAIModel as OpenAIChatModel
try:
    from pydantic_ai.providers.ollama import OllamaProvider
except ImportError:
    # Fallback: use OpenAIModel with Ollama endpoint directly
    from pydantic_ai.models.openai import OpenAIModel


    class OllamaProvider:
        def __init__(self, base_url: str, api_key: str = None):
            self.base_url = base_url
            self.api_key = api_key

        def __call__(self, model_name: str, **kwargs):
            # Return an OpenAIModel configured for Ollama
            return OpenAIModel(
                model_name=model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                **kwargs,
            )

from src.clawflow.agents.agent import (
    IntentModel,
    PlanModel,
    ReflectModel,
    AgentState,
    hash_step,
    async_invoke_chat_with_retry,
    current_step,
)
from const.const import (
    OLLAMA_BASE_URL,
    MAX_ITERATIONS,
    MAX_TOOL_CONSECUTIVE_COUNT,
)
from pydantic import BaseModel, Field
from src.clawflow.skills.skills_registry import REGISTRY as SKILL_REGISTRY
from my_tools import CLIENT_MANAGER
from src.clawflow.utils.utils import (
    normalize_provider,
    truncate,
    msg_preview,
    extract_resp_id,
    extract_finish_reason,
    extract_text_content,
    try_parse_json,
    to_dict,
    ai_meta,
    tool_calls_signature,
    text_hash,
    retry_with_backoff,
    extract_tool_calls, log_request, log_response,
)

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# ---------------------------
# Context pruning knobs
# ---------------------------
LLM_STEP_HISTORY_MAX_MSGS = int(env("LLM_STEP_HISTORY_MAX_MSGS", "12") or 12)
LLM_TOOL_MSG_MAX_CHARS = int(env("LLM_TOOL_MSG_MAX_CHARS", "4000") or 4000)
LLM_DEP_OUTPUT_MAX_CHARS = int(env("LLM_DEP_OUTPUT_MAX_CHARS", "1500") or 1500)
LLM_INTENT_HISTORY_PREVIEW_MSGS = int(env("LLM_INTENT_HISTORY_PREVIEW_MSGS", "8") or 8)

_FINAL_STEP_KEYWORDS = ("最终", "总结", "汇总", "结论", "最终答案", "final", "summary", "report", "deliverable")

# Global tool registry populated by `init_mcp_tools()`.
# NOTE: must be a real dict (not `typing.Dict[...]`).
MCP_TOOLS_REGISTRY: Dict[str, Any] = {}


def _coerce_step_dict(step: Any, *, default_id: str) -> tuple[dict, bool, str]:
    """将单个 step 归一化为 dict，避免出现 str/模型对象导致 `.get` 崩溃。"""
    changed = False
    reason = ""
    s = step

    # Pydantic model / LC structured obj
    if hasattr(s, "model_dump"):
        try:
            s = s.model_dump()
            changed = True
            reason = "model_dump"
        except Exception:
            s = step

    # JSON string step
    if isinstance(s, str):
        parsed = try_parse_json(s)
        if isinstance(parsed, dict):
            s = parsed
            changed = True
            reason = "str_json"
        else:
            # 非 JSON 的字符串：包装成可执行 step
            s = {
                "id": default_id,
                "title": truncate(str(step), 40),
                "agent": "writer",
                "task": str(step),
                "acceptance": "输出结果",
                "inputs": {},
                "depends_on": [],
            }
            changed = True
            reason = "str_wrapped"

    if not isinstance(s, dict):
        return {
            "id": default_id,
            "title": "",
            "agent": "writer",
            "task": "",
            "acceptance": "",
            "inputs": {},
            "depends_on": [],
        }, True, f"not_dict:{type(s).__name__}"

    # Ensure minimal keys
    if not str(s.get("id") or "").strip():
        s["id"] = default_id
        changed = True
        reason = reason or "missing_id"
    if not str(s.get("agent") or "").strip():
        s["agent"] = "writer"
        changed = True
        reason = reason or "missing_agent"
    if not isinstance(s.get("inputs"), dict):
        s["inputs"] = {}
        changed = True
        reason = reason or "bad_inputs"
    if not isinstance(s.get("depends_on"), list):
        s["depends_on"] = []
        changed = True
        reason = reason or "bad_depends_on"

    return s, changed, reason


def _coerce_steps_list(steps: Any) -> tuple[list[dict], bool, str]:
    """将 plan['steps'] 归一化为 list[dict]。"""
    if steps is None:
        return [], False, ""

    raw = steps
    changed = False
    reason = ""

    # steps 可能被错误写成 JSON 字符串
    if isinstance(raw, str):
        parsed = try_parse_json(raw)
        if isinstance(parsed, dict) and "steps" in parsed:
            raw = parsed.get("steps")
            changed = True
            reason = "steps_from_json_obj"
        elif isinstance(parsed, list):
            raw = parsed
            changed = True
            reason = "steps_from_json_list"
        else:
            return [], True, "steps_was_str"

    if not isinstance(raw, list):
        return [], True, f"steps_not_list:{type(raw).__name__}"

    out: list[dict] = []
    for i, st in enumerate(raw):
        s2, ch2, r2 = _coerce_step_dict(st, default_id=f"step{i + 1}")
        if ch2:
            changed = True
            reason = reason or r2
        out.append(s2)

    return out, changed, reason


def _normalize_plan_in_state(state: AgentState) -> tuple[AgentState, bool, str]:
    """保证 state['plan']['steps'] 永远是 list[dict]，必要时回写 state。"""
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return state, False, ""

    steps_norm, changed, reason = _coerce_steps_list(plan.get("steps"))
    if not changed:
        return state, False, ""

    plan2 = dict(plan)
    plan2["steps"] = steps_norm
    state2 = dict(state)
    state2["plan"] = plan2
    return state2, True, reason


def looks_like_final_step(step: Dict[str, Any]) -> bool:
    title = str(step.get("title") or "")
    task = str(step.get("task") or "")
    acceptance = str(step.get("acceptance") or "")
    s = f"{title} {task} {acceptance}".lower()
    return any(k.lower() in s for k in _FINAL_STEP_KEYWORDS)


def tool_result_to_text(res: Any) -> str:
    try:
        if isinstance(res, (dict, list)):
            s = json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        else:
            s = str(res)
    except Exception:
        s = repr(res)
    return truncate(s, LLM_TOOL_MSG_MAX_CHARS)


def normalize_intent_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    # constraints must be List[str]
    c = d.get("constraints")
    if c is None:
        d["constraints"] = []
    elif isinstance(c, list):
        d["constraints"] = [str(x) for x in c if x is not None and str(x).strip()]
    elif isinstance(c, dict):
        d["constraints"] = [f"{k}: {v}" for k, v in c.items()]
    else:
        d["constraints"] = [str(c)]

    # entities drop empty
    ents = d.get("entities") or []
    if isinstance(ents, list):
        d["entities"] = [e for e in ents if isinstance(e, dict) and str(e.get("value") or "").strip()]
    else:
        d["entities"] = []

    # role_preset ensure dict
    rp = d.get("role_preset")
    if rp is None or not isinstance(rp, dict):
        d["role_preset"] = {}

    return d


def step_local_messages(state: AgentState, step_id: str) -> tuple[list[BaseMessage], dict[str, int]]:
    msgs_all: list[BaseMessage] = list(state.get("messages") or [])
    cursors: dict[str, int] = dict(state.get("step_msg_start") or {})

    # First time we enter a step -> set cursor to current end (empty local history)
    if step_id and step_id not in cursors:
        cursors[step_id] = len(msgs_all)

    start = int(cursors.get(step_id, 0) or 0)
    step_msgs = msgs_all[start:]

    # cap by message count
    if len(step_msgs) > LLM_STEP_HISTORY_MAX_MSGS:
        step_msgs = step_msgs[-LLM_STEP_HISTORY_MAX_MSGS:]

    return step_msgs, cursors


logger = logging.getLogger(__name__)


def log_event(level: int, event: str, **fields):
    # Use unified key=value output (keep it on one line when possible)
    kv = " ".join(
        [f'{k}="{truncate(v, 600)}"' for k, v in fields.items() if v is not None]
    )
    logger.log(level, f"{event} {kv}".rstrip())


def has_orphan_tool_message(msgs: list[BaseMessage]) -> bool:
    seen_tool_calls = False
    for m in msgs:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            seen_tool_calls = True
        if isinstance(m, ToolMessage) and not seen_tool_calls:
            return True
    return False


@utils.timer
async def async_invoke_structured_with_retry(
        llm, msgs, *, role="unknown", retries=3, base_sleep=0.5
):
    attempt = 0

    @utils.timer
    @retry_with_backoff(retries=retries, base_sleep=base_sleep)
    async def _invoke_once():
        nonlocal attempt
        attempt += 1
        try_i = attempt - 1

        log_event(
            logging.INFO,
            "llm.structured.call",
            try_i=try_i,
            role=role,
            msgs=msg_preview(msgs),
        )

        if has_orphan_tool_message(msgs):
            log_event(
                logging.INFO,
                "orphan_tool_message",
                try_i=try_i,
                role=role,
                msg_preview=msg_preview(msgs, n=6),
            )

        # Automatically inject time context
        msgs_with_time = utils.inject_time_context(msgs)
        if msgs_with_time != msgs:
            log_event(
                logging.DEBUG,
                "llm.structured.time_injected",
                try_i=try_i,
                role=role,
                original_len=len(msgs),
                new_len=len(msgs_with_time),
            )

        msgs_with_time = utils.filter_messages_for_llm(msgs_with_time)
        start = time.perf_counter()

        answer = await llm.ainvoke(msgs_with_time)
        end = time.perf_counter()
        logger.info(f"{__name__} 耗时: {end - start:.6f} 秒")

        usage = ai_meta(answer).get("usage_metadata", {})
        # Add usage info to the answer
        if isinstance(answer, dict):
            answer["usage_metadata"] = usage

        # Diagnostic log: record basic answer structure (without leaking full model content)
        if isinstance(answer, dict):
            log_event(
                logging.DEBUG,
                "llm.structured.answer_dict",
                try_i=try_i,
                role=role,
                keys=list(answer.keys()),
                parsed=answer.get("parsed"),
                parsing_error=answer.get("parsing_error"),
            )
            raw = answer.get("raw")
            if raw:
                finish_reason = extract_finish_reason(raw)
                tool_calls = extract_tool_calls(raw)
                invalid_tool_calls = getattr(raw, "additional_kwargs", {}).get(
                    "invalid_tool_calls"
                )
                log_event(
                    logging.DEBUG,
                    "llm.structured.raw_meta",
                    try_i=try_i,
                    role=role,
                    finish_reason=finish_reason,
                    tool_calls_cnt=len(tool_calls),
                    invalid_tool_calls_cnt=(
                        len(invalid_tool_calls) if invalid_tool_calls else 0
                    ),
                )
        else:
            log_event(
                logging.DEBUG,
                "llm.structured.answer_type",
                try_i=try_i,
                role=role,
                type=type(answer).__name__,
            )

        # Always log minimal meta; never log full model content (avoid leaking "thinking"/raw output)
        _meta_obj = answer
        if isinstance(answer, dict) and answer.get("raw") is not None:
            _meta_obj = answer.get("raw")
        resp_id = extract_resp_id(_meta_obj)
        finish_reason = extract_finish_reason(_meta_obj)
        tool_calls = extract_tool_calls(_meta_obj)
        log_event(
            logging.INFO,
            "llm.chat.meta",
            role=role,
            resp_id=resp_id,
            finish_reason=finish_reason,
            tool_calls_cnt=len(tool_calls),
            usage=json.dumps(usage, ensure_ascii=False),
        )

        if isinstance(answer, dict) and answer.get("parsed") is None:
            raw = answer.get("raw")
            # Prefer raw.content as the "real" model output
            raw_text = getattr(raw, "content", None)

            if not raw_text:
                raw_text = extract_text_content(raw) or str(raw)
                # If still empty, check invalid_tool_calls
                if not raw_text:
                    invalid_tc = getattr(raw, "additional_kwargs", {}).get(
                        "invalid_tool_calls"
                    )
                    if invalid_tc:
                        raw_text = json.dumps(
                            {"invalid_tool_calls": invalid_tc}, ensure_ascii=False
                        )

            # Fallback: parse JSON ourselves in case Ollama/adapter doesn't set parsed
            parsed = try_parse_json(raw_text)

            # If raw_text parsing fails, check invalid_tool_calls and try to extract arguments
            if parsed is None:
                invalid_tc = getattr(raw, "additional_kwargs", {}).get(
                    "invalid_tool_calls"
                )
                if invalid_tc:
                    log_event(
                        logging.WARN,
                        "llm.structured.invalid_tool_calls",
                        try_i=try_i,
                        role=role,
                        invalid_tool_calls=json.dumps(invalid_tc, ensure_ascii=False),
                    )
                    # Iterate invalid tool calls and try to parse the arguments field
                    for call in invalid_tc:
                        if isinstance(call, dict):
                            # Log detailed structure for debugging
                            log_event(
                                logging.DEBUG,
                                "llm.structured.invalid_call_detail",
                                try_i=try_i,
                                role=role,
                                call=json.dumps(call, ensure_ascii=False),
                            )
                            # OpenAI format: function.arguments
                            func = call.get("function", {})
                            if isinstance(func, dict):
                                args_str = func.get("arguments")
                                if isinstance(args_str, str) and args_str:
                                    log_event(
                                        logging.DEBUG,
                                        "llm.structured.parsing_attempt",
                                        try_i=try_i,
                                        role=role,
                                        args_str=args_str[:200],
                                    )
                                    parsed = try_parse_json(args_str)
                                    if parsed is not None:
                                        log_event(
                                            logging.INFO,
                                            "llm.structured.recovered_from_invalid",
                                            try_i=try_i,
                                            role=role,
                                            parsed_type=type(parsed).__name__,
                                        )
                                        break
                                elif isinstance(args_str, dict):
                                    # arguments is already a dict
                                    parsed = args_str
                                    log_event(
                                        logging.INFO,
                                        "llm.structured.recovered_from_invalid_dict",
                                        try_i=try_i,
                                        role=role,
                                    )
                                    break
                            # Other formats: use arguments field directly
                            args_str = call.get("arguments")
                            if isinstance(args_str, str) and args_str:
                                log_event(
                                    logging.DEBUG,
                                    "llm.structured.parsing_attempt_direct",
                                    try_i=try_i,
                                    role=role,
                                    args_str=args_str[:200],
                                )
                                parsed = try_parse_json(args_str)
                                if parsed is not None:
                                    log_event(
                                        logging.INFO,
                                        "llm.structured.recovered_from_invalid_direct",
                                        try_i=try_i,
                                        role=role,
                                    )
                                    break
                            elif isinstance(args_str, dict):
                                parsed = args_str
                                log_event(
                                    logging.INFO,
                                    "llm.structured.recovered_from_invalid_direct_dict",
                                    try_i=try_i,
                                    role=role,
                                )
                                break

            if parsed is not None:
                answer["parsed"] = parsed
                answer["parsing_error"] = None

                log_event(
                    logging.WARN,
                    "llm.structured.fallback_parsed",
                    try_i=try_i,
                    role=role,
                    preview=str(parsed),
                )

                log_event(logging.WARN, "llm.structured.ok", try_i=try_i, role=role)

                return answer

            # If parsed is still None, provide role-based defaults
            log_event(
                logging.WARN,
                "llm.structured.parsed_still_none",
                try_i=try_i,
                role=role,
                raw_text=raw_text,
            )
            if role == "intent":
                parsed = {
                    "task_type": "other",
                    "user_goal": "未识别的用户请求",
                    "domains": [],
                    "deliverable": "direct_answer",
                    "entities": [],
                    "need_web": False,
                    "suggested_tools": [],
                    "constraints": [],
                    "missing_info": [],
                    "output_language": "zh",
                    "role_preset": {},
                }
            elif role == "plan":
                user_req = "未知请求"
                # Try to extract the user request from messages
                for msg in msgs:
                    if isinstance(msg, HumanMessage):
                        content = getattr(msg, "content", "")
                        if "用户需求" in content:
                            # Simple extraction
                            user_req = content.split("用户需求：")[-1][:200]
                            break
                parsed = {
                    "version": 1,
                    "objective": user_req,
                    "steps": [
                        {
                            "id": "s1",
                            "title": "直接回答",
                            "agent": "writer",
                            "task": f"直接回答用户问题：{user_req}",
                            "acceptance": "回答清晰、正确、无多余废话",
                            "inputs": {},
                            "depends_on": [],
                        }
                    ],
                }
            elif role == "reflect":
                parsed = {
                    # Safety first: don't pass if reflect parsing fails; otherwise empty/bad artifacts may be treated as "accepted"
                    "decision": "retry",
                    "reason": f"LLM 输出解析失败，回退重试: {raw_text[:100]}",
                    "required_changes": [
                        "必须只输出一个 JSON 对象（不要 markdown/解释文字）",
                        "严格符合 ReflectModel schema（decision/reason/required_changes/plan_patch）",
                    ],
                    "plan_patch": {"version": 1, "objective": "", "steps": []},
                }
            elif role == "skill_router":
                parsed = {"selected_skill_ids": []}
            else:
                parsed = {"error": "无法解析模型输出", "raw": raw_text}

            answer["parsed"] = parsed
            answer["parsing_error"] = None
            log_event(
                logging.WARN,
                "llm.structured.default_parsed",
                try_i=try_i,
                role=role,
                preview=str(parsed)[:200],
            )
            return answer

        log_event(
            logging.INFO,
            "llm.structured.ok",
            try_i=try_i,
            role=role,
            usage=json.dumps(usage, ensure_ascii=False),
        )

        return answer

    return await _invoke_once()


def mk_chat_ollama(
        *,
        model: str,
        base_url: str,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        timeout_s: int,
) -> BaseChatModel:
    if ChatOllama is None:
        raise RuntimeError("langchain_ollama is not installed.")

    # Return instance directly; do not call .bind()
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        keep_alive="0",
        # 通用参数（sync/async 都会用）
        client_kwargs={
            "timeout": httpx.Timeout(60.0, connect=10.0),
        },
        # 同步专用 hooks
        sync_client_kwargs={
            "event_hooks": {
                "request": [log_request],
                "response": [log_response],
            }
        },
        # 异步专用 hooks（你如果用 ainvoke/astream 就靠这个）
        async_client_kwargs={
            "event_hooks": {
                "request": [log_request],
                "response": [log_response],
            }
        },
        # Note: if you want to ensure non-streaming output, invoke() is usually non-streaming by default,
        # or you can pass it explicitly, e.g. model.invoke(..., stream=False)
    )


def ensure_v1(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    return base_url if base_url.endswith("/v1") else base_url + "/v1"


def mk_agent_ollama(
        *, model: str, temperature: float, num_ctx: int, num_predict: int, timeout_s: int
) -> Agent:
    base_url = ensure_v1(OLLAMA_BASE_URL)

    provider = OllamaProvider(
        base_url=base_url,
        api_key=None,  # Can be None locally; required in cloud (or via env vars)
    )

    # Agent-level default settings: temperature, timeout, max output tokens
    settings = ModelSettings(
        temperature=temperature,
        max_tokens=num_predict if num_predict and num_predict > 0 else None,
        timeout=float(timeout_s),
        # Compatibility fallback: some OpenAI SDK wrappers use max_completion_tokens; Ollama docs use max_tokens.
        # Put max_tokens into request body explicitly to ensure num_predict takes effect.
        extra_body=(
            {"max_tokens": int(num_predict)} if num_predict and num_predict > 0 else {}
        ),
    )

    # num_ctx: Ollama's OpenAI-compatible API usually can't change context length per request; it's typically a model/Modelfile-level setting (see below).
    _ = num_ctx  # Keep signature; if you must control num_ctx dynamically, see "what to do with num_ctx" below

    llm = OpenAIChatModel(
        model_name=model,
        provider=provider,
    )

    return Agent(llm, model_settings=settings)


def mk_chat_openai_compact(
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
        extra_body: Dict[str, Any] | None = None,
        streaming: bool = False,
) -> BaseChatModel:
    if ChatOpenAI is None:
        raise RuntimeError(
            "langchain_openai is not installed. Please `pip install langchain-openai`."
        )
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    # Compatibility notes:
    # - `langchain_openai.ChatOpenAI` has historically changed init kwarg names across versions.
    # - We prefer a single-pass construction based on the current signature.
    # - If it still fails (unexpected wrapper differences), fall back to the old multi-candidate try.
    sig = inspect.signature(ChatOpenAI.__init__)
    params = sig.parameters

    kw: Dict[str, Any] = {}

    # Model name kwarg
    if "model" in params:
        kw["model"] = model
    elif "model_name" in params:
        kw["model_name"] = model
    else:
        # As a last resort, keep the historical name; fallback handles TypeError.
        kw["model"] = model

    # Common sampling / limit args
    if "temperature" in params:
        kw["temperature"] = temperature
    if "max_tokens" in params:
        kw["max_tokens"] = max_tokens

    # Provider-specific extensions (e.g., DashScope/Qwen thinking)
    # `langchain_openai.ChatOpenAI` supports `extra_body` in recent versions; keep best-effort.
    if extra_body is not None:
        if "extra_body" in params:
            kw["extra_body"] = extra_body
        elif "model_kwargs" in params:
            # Legacy adapters sometimes accept `model_kwargs`
            kw["model_kwargs"] = dict(extra_body)

    # Streaming toggle (some reasoning/thinking modes require streaming)
    if "streaming" in params:
        kw["streaming"] = streaming
    elif "stream" in params:
        kw["stream"] = streaming

    # Auth & base url kwargs
    if api_key:
        if "api_key" in params:
            kw["api_key"] = api_key
        elif "openai_api_key" in params:
            kw["openai_api_key"] = api_key

    if base_url:
        if "base_url" in params:
            kw["base_url"] = base_url
        elif "openai_api_base" in params:
            kw["openai_api_base"] = base_url

    # Timeout kwarg name varies
    if "timeout" in params:
        kw["timeout"] = timeout_s
    elif "request_timeout" in params:
        kw["request_timeout"] = timeout_s

    if "max_retries" in params:
        kw["max_retries"] = 2

    kw = {k: v for k, v in kw.items() if v is not None}

    candidates = [
        # Legacy fallback for older/newer langchain_openai wrappers
        dict(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
            max_retries=2,
        ),
        dict(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout_s,
            max_retries=2,
        ),
        dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
            max_retries=2,
        ),
        dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout_s,
            max_retries=2,
        ),
    ]

    http_async_client = httpx.AsyncClient(
        timeout=60.0,
        event_hooks={"request": [log_request], "response": [log_response]},
    )

    http_client = httpx.Client(
        timeout=60.0,
        event_hooks={"request": [log_request], "response": [log_response]},
    )

    # First try: signature-adaptive kwargs
    try:
        return ChatOpenAI(
            **kw,
            http_async_client=http_async_client,
            http_client=http_client,
        )
    except TypeError as first_err:
        # Fallback: multi-candidate try (historical behavior)
        last: Exception | None = first_err
        for cand in candidates:
            try:
                return ChatOpenAI(
                    **{k: v for k, v in cand.items() if v is not None},
                    http_async_client=http_async_client,
                    http_client=http_client,
                )
            except TypeError as e:
                last = e
                continue
        raise last or RuntimeError("Failed to construct ChatOpenAI")


def mk_chat_google_genai(
        *,
        model: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
        timeout_s: int,
) -> BaseChatModel:
    if ChatGoogleGenerativeAI is None or not callable(ChatGoogleGenerativeAI):
        detail = ""
        if _GEMINI_IMPORT_ERROR is not None:
            detail = f" | import_error={type(_GEMINI_IMPORT_ERROR).__name__}: {_GEMINI_IMPORT_ERROR}"
        raise RuntimeError(
            "Gemini provider is not available: `langchain_google_genai.ChatGoogleGenerativeAI` not loaded. "
            "Install/repair deps via `pip install langchain-google-genai` (and ensure compatible langchain_core)."
            + detail
        )

    # Compatibility notes:
    # - `ChatGoogleGenerativeAI` init kwargs may differ across versions.
    # - Prefer signature-adaptive kwargs (similar to mk_chat_openai_compact).
    sig = inspect.signature(ChatGoogleGenerativeAI.__init__)
    params = sig.parameters

    kw: Dict[str, Any] = {}

    # Model name kwarg
    if "model" in params:
        kw["model"] = model
    elif "model_name" in params:
        kw["model_name"] = model
    else:
        kw["model"] = model

    # API key kwarg name varies; also set env as fallback
    if api_key:
        if "google_api_key" in params:
            kw["google_api_key"] = api_key
        elif "api_key" in params:
            kw["api_key"] = api_key
        os.environ.setdefault("GOOGLE_API_KEY", api_key)

    if "temperature" in params:
        kw["temperature"] = temperature

    # Token limit kwarg name varies
    if "max_output_tokens" in params:
        kw["max_output_tokens"] = max_tokens
    elif "max_tokens" in params:
        kw["max_tokens"] = max_tokens

    # Timeout kwarg name varies; if unsupported, ignore
    if "timeout" in params:
        kw["timeout"] = timeout_s
    elif "request_timeout" in params:
        kw["request_timeout"] = timeout_s

    kw = {k: v for k, v in kw.items() if v is not None}
    return ChatGoogleGenerativeAI(**kw)


def mk_role_chat(role: str) -> BaseChatModel:
    # Support per-role provider config; fall back to shared config
    provider = normalize_provider(env(f"{role.upper()}.PROVIDER"))
    if provider is None or provider == "":
        raise RuntimeError(f"{role.upper()}.PROVIDER not found.")

    # Read provider-specific defaults from config
    key_model = f"{role.upper()}.MODEL"
    model = env(key_model)
    key_base_url = f"LLM.PROVIDERS.{provider.upper()}.BASE_URL"
    base_url = env(key_base_url)
    key_api_key = f"LLM.PROVIDERS.{provider.upper()}.API_KEY"
    api_key = env(key_api_key)

    if model is None or model == "":
        raise RuntimeError(f"Model {key_model} not found.")
    # Gemini（Google AI Studio）不需要 base_url；其他 OpenAI-compatible provider 仍校验
    if provider not in ("ollama", "gemini") and (base_url is None or base_url == ""):
        raise RuntimeError(f"Base URL {key_base_url} not found.")
    if provider.upper() != "OLLAMA" and (api_key is None or api_key == ""):
        raise RuntimeError(f"API key {key_api_key} not found.")

    # NOTE:
    # - OpenAI-compatible providers use `max_tokens`.
    # - Ollama uses `num_predict` (hard cap on generated tokens). Some reasoning models
    #   may spend many tokens before emitting a final JSON, so give structured nodes more headroom.
    if provider == "ollama":
        if role == "intent":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.1,
                2048,
                8192,
                1024,
                180,
            )
        elif role == "planner":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.1,
                2048,
                8192,
                1024,
                180,
            )
        elif role == "agent":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.3,
                4096,
                8192,
                2048,
                240,
            )
        elif role == "reflector":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.0,
                2048,
                8192,
                1024,
                180,
            )
        else:
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.0,
                2048,
                6144,
                1024,
                180,
            )
    else:
        if role == "intent":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.1,
                512,
                2048,
                512,
                120,
            )
        elif role == "planner":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.1,
                512,
                2048,
                512,
                120,
            )
        elif role == "agent":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.3,
                1536,
                4096,
                1536,
                600,
            )
        elif role == "reflector":
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.0,
                512,
                3072,
                512,
                300,
            )
        else:
            temperature, max_tokens, num_ctx, num_predict, timeout_s = (
                0.0,
                512,
                3072,
                512,
                300,
            )

    if provider == "ollama":
        return mk_chat_ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            timeout_s=timeout_s,
        )
    if provider == "gemini":
        return mk_chat_google_genai(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    # DashScope/Qwen: thinking 模式与 tool_choice/object（function calling）存在兼容性限制。
    # 经验法则：
    # - agent（会用工具的节点）可以开启 thinking，但不要强制 tool_choice（保持 auto）。
    # - intent/planner/reflector（structured 输出节点）必须显式关闭 thinking，避免 400 InvalidParameter。
    extra_body: Dict[str, Any] | None = None
    # DashScope/Qwen:
    # - thinking 模式常见要求：stream=true（否则可能报 non-streaming + thinking 的参数错误）
    # - thinking 模式不支持 tool_choice=object/required（function calling 强制工具）
    # 因此：这里默认对 DashScope 开启 streaming；structured 节点尽量关闭 thinking。
    streaming_flag: bool = False
    if isinstance(base_url, str) and "dashscope.aliyuncs.com" in base_url:
        streaming_flag = True  # 更稳：即使 thinking 为默认开启，也不会因 stream=false 触发额外 400
        if role in ("intent", "planner", "reflector"):
            extra_body = {"enable_thinking": False}
        elif role == "agent":
            extra_body = {"enable_thinking": True}

    return mk_chat_openai_compact(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        extra_body=extra_body,
        streaming=streaming_flag,
    )


def with_structured(llm: Runnable, schema_model: Any) -> Runnable:
    """
    Return a wrapper that produces `schema_model` and returns a dict shaped like
    LangChain's `with_structured_output(..., include_raw=True)`:

        {"raw": AIMessage, "parsed": <schema_model>|None, "parsing_error": str|None}

    Why this exists (DashScope/Qwen):
    - DashScope "thinking mode" rejects `tool_choice` when it is set to an object/required.
    - LangChain's `method="function_calling"` forces `tool_choice={"type":"function",...}`.
    - Therefore for DashScope we MUST avoid function calling for structured outputs, and instead:
        (1) prompt for strict JSON-only output
        (2) parse JSON locally into the pydantic model

    For Ollama/Gemini we keep the original behavior.
    """

    # Provider detection helpers (robust across langchain_openai versions)
    def _get_base_url(x: Any) -> str:
        for attr in ("base_url", "openai_api_base"):
            v = getattr(x, attr, None)
            if isinstance(v, str) and v:
                return v
        for c_attr in ("client", "_client", "async_client", "_async_client"):
            c = getattr(x, c_attr, None)
            if c is None:
                continue
            v = getattr(c, "base_url", None)
            if v:
                return str(v)
        # Fallback: env that mk_chat_openai_compact sets
        return os.environ.get("OPENAI_BASE_URL", "") or ""

    def _is_dashscope(x: Any) -> bool:
        bu = _get_base_url(x)
        return isinstance(bu, str) and ("dashscope.aliyuncs.com" in bu)

    def _schema_json(sm: Any) -> dict:
        # pydantic v2
        if hasattr(sm, "model_json_schema"):
            return sm.model_json_schema()
        # pydantic v1
        if hasattr(sm, "schema"):
            return sm.schema()
        return {}

    def _validate(sm: Any, obj: Any):
        # pydantic v2
        if hasattr(sm, "model_validate"):
            return sm.model_validate(obj)
        # pydantic v1
        if hasattr(sm, "parse_obj"):
            return sm.parse_obj(obj)
        # Fallback
        return sm(**obj)

    # Best-effort detect Ollama/Gemini.
    is_ollama = (ChatOllama is not None) and isinstance(llm, ChatOllama)
    is_gemini = (ChatGoogleGenerativeAI is not None) and isinstance(llm, ChatGoogleGenerativeAI)

    if is_ollama:
        try:
            llm = llm.bind(format="json")
        except Exception:
            pass

        for m in get_args(Literal["json_mode", "json_schema"]):
            try:
                return llm.with_structured_output(schema_model, method=m, include_raw=True)
            except Exception:
                continue
        return llm.with_structured_output(schema_model, include_raw=True)

    if is_gemini:
        try:
            return llm.with_structured_output(schema_model, method="json_schema", include_raw=True)
        except Exception:
            return llm.with_structured_output(schema_model, include_raw=True)

    # DashScope/Qwen: avoid function_calling (tool_choice object) completely.
    if _is_dashscope(llm):
        schema = _schema_json(schema_model)
        schema_str = json.dumps(schema, ensure_ascii=False)

        json_only_system = (
            "你是一个严格的 JSON 生成器。\n"
            "只输出一个 JSON 对象，禁止输出任何解释、Markdown、代码块、前后缀文本。\n"
            "JSON 必须能被标准 JSON 解析器直接解析。\n"
            "请严格遵循下面的 JSON Schema（字段名/类型/必填）：\n"
            f"{schema_str}"
        )

        class _DashScopeJSONStructuredWrapper:
            def __init__(self, base_llm):
                self._base = base_llm

            def invoke(self, msgs, **kwargs):
                # Sync fallback
                raw = self._base.invoke([SystemMessage(content=json_only_system), *msgs])
                return self._wrap(raw)

            async def ainvoke(self, msgs, **kwargs):
                raw = await self._base.ainvoke([SystemMessage(content=json_only_system), *msgs])
                return self._wrap(raw)

            def _wrap(self, raw_msg: Any) -> dict:
                text = extract_text_content(raw_msg)
                parsed = None
                err = None
                try:
                    obj = try_parse_json(text)
                    if obj is None:
                        # Best-effort: extract first {...} block
                        m = re.search(r"\{[\s\S]*\}", text)
                        if m:
                            obj = json.loads(m.group(0))
                    if obj is None:
                        raise ValueError("failed to parse json from model output")
                    parsed = _validate(schema_model, obj)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                return {"raw": raw_msg, "parsed": parsed, "parsing_error": err}

        return _DashScopeJSONStructuredWrapper(llm)

    # Non-DashScope OpenAI-compatible: try function calling first.
    try:
        return llm.with_structured_output(schema_model, method="function_calling", include_raw=True)
    except Exception:
        try:
            return llm.with_structured_output(schema_model, method="json_schema", include_raw=True)
        except Exception:
            return llm.with_structured_output(schema_model, include_raw=True)


@utils.timer
async def tool_invoke(tool_obj: Any, args: Dict[str, Any]) -> Any:
    """Invoke tools via various LangChain-compatible interfaces."""
    logger.debug(
        f"DEBUG: Invoking tool {type(tool_obj)} with methods: {dir(tool_obj)[:5]}..."
    )

    if args is None:
        args = {}

    # 1. Prefer async invocation
    if hasattr(tool_obj, "ainvoke"):
        return await tool_obj.ainvoke(args)

    # 2. Standard invoke (recommended)
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(args)

    # 3. Low-level func (StructuredTool)
    if hasattr(tool_obj, "func"):
        try:
            return tool_obj.func(**args)
        except TypeError:
            return tool_obj.func(args)

    # 4. Legacy run
    if hasattr(tool_obj, "run"):
        try:
            return tool_obj.run(**args)
        except TypeError:
            return tool_obj.run(args)

    # 5. Callable object
    if callable(tool_obj):
        return tool_obj(**args)

    logger.error(f"ERROR: {type(tool_obj)} is not invokable.")

    raise TypeError(f"Tool {tool_obj!r} is not invokable.")


@utils.timer
async def safe_tool_node(state: AgentState) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    if not msgs:
        return {}

    last_msg = msgs[-1]
    tool_calls = extract_tool_calls(last_msg) or []
    if not tool_calls:
        return {}

    # === Key: limit tool_call count per iteration (default 1) ===
    max_calls = int(env("MAX_TOOL_CALLS_PER_TURN", "1"))
    tool_calls = tool_calls[:max(1, max_calls)]

    new_msgs: List[BaseMessage] = []
    for idx, call in enumerate(tool_calls):
        name = str(call.get("name") or "").strip()
        call_id = str(call.get("id") or f"call_{name}_{idx}")

        # args compatibility: dict or JSON string
        raw_args = call.get("args")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        tool = MCP_TOOLS_REGISTRY.get(name) or MCP_TOOLS_REGISTRY.get(name.lower())
        if not tool:
            content = tool_result_to_text({
                "error": "tool_not_found",
                "tool_name": name,
                "known_tools": sorted(list(MCP_TOOLS_REGISTRY.keys())),
            })
            new_msgs.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
            continue

        try:
            res = await tool_invoke(tool, args)
            content = tool_result_to_text(res)
        except Exception as e:
            content = tool_result_to_text({
                "error": "tool_execution_failed",
                "tool_name": name,
                "exception": f"{type(e).__name__}: {e}",
            })

        if os.getenv("AGENT_LOG_LLM_DUMP", "0") == "1":
            try:
                print("--------- tool call ---------")
                print(f"name={name} call_id={call_id}")
                print("--------- tool args ---------")
                print(json.dumps(args, ensure_ascii=False, indent=2, default=str))
                print("--------- tool result ---------")
                # Preview results too, to avoid flooding the console
                preview = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
                print(preview[:2000])
            except Exception:
                pass

        new_msgs.append(ToolMessage(content=content, tool_call_id=call_id, name=name))

    return {"messages": new_msgs}


def _is_module_spec(s: str) -> bool:
    """Return True if s looks like a Python module path or an explicit module: spec."""
    t = (s or "").strip()
    if not t:
        return False
    if t.startswith("module:"):
        return True
    # Heuristic: dotted module path, not a .py file
    if t.endswith(".py"):
        return False
    if "/" in t or "\\" in t:
        return False
    return "." in t


def _my_tools_pkg_dir() -> Path:
    """Return installed my_tools package directory."""
    import importlib

    pkg = importlib.import_module("my_tools")
    p = Path(getattr(pkg, "__file__", "") or "").resolve()
    if not p.exists():
        raise RuntimeError("无法定位已安装的 my_tools 包目录（my_tools.__file__ 不存在）")
    return p.parent


def _resolve_mcp_entry(entry: str) -> str:
    """Resolve MCP server entry to either a script path or a module spec.

    支持三种写法：
    - module 模式："module:my_tools.mcp_ro" 或 "my_tools.mcp_ro" -> 规范化为 "module:..."
    - 脚本绝对路径："/abs/path/mcp_ro.py" -> 原样返回
    - 脚本相对路径/文件名："mcp_ro.py" 或 "subdir/mcp_ro.py" -> 从已安装的 my_tools 包目录解析
    """
    e = (entry or "").strip()
    if not e:
        return e

    # module mode
    if _is_module_spec(e):
        return e if e.startswith("module:") else f"module:{e}"

    # script mode
    p = Path(e).expanduser()
    if p.is_absolute():
        return str(p)

    # Relative script: resolve under installed my_tools package directory
    base = _my_tools_pkg_dir()
    return str((base / p).resolve())


def update_global_registry(tools: List[StructuredTool]):
    """
    Update the global MCP_TOOLS_REGISTRY in tools.py with the given tools.
    """
    for t in tools:
        MCP_TOOLS_REGISTRY[t.name] = t
        # Also add sanitized names if needed, similar to build_tool_registry
        MCP_TOOLS_REGISTRY[t.name.strip()] = t
        MCP_TOOLS_REGISTRY[t.name.strip().lower()] = t


async def init_mcp_tools() -> Dict[str, List]:
    """
    Initialize MCP tools and return grouped tool sets.
    Uses temporary sessions to avoid zombie processes during initialization.
    """
    ro_script = env("MCP.SERVERS.RO", "my_tools.mcp_ro")
    if not ro_script:
        raise ValueError("MCP.SERVERS.RO environment variable is required")

    exec_script = env("MCP.SERVERS.EXEC", "my_tools.mcp_exec_safe")
    if not exec_script:
        raise ValueError("MCP.SERVERS.EXEC environment variable is required")

    artifacts_script = env("MCP.SERVERS.ARTIFACTS", "my_tools.mcp_artifacts")
    if not artifacts_script:
        raise ValueError("MCP.SERVERS.ARTIFACTS environment variable is required")

    net_script = env("MCP.SERVERS.NET", "my_tools.mcp_net")
    if not net_script:
        raise ValueError("MCP.SERVERS.NET environment variable is required")

    shell_script = env("MCP.SERVERS.SHELL", "my_tools.mcp_shell")
    if not shell_script:
        raise ValueError("MCP.SERVERS.SHELL environment variable is required")

    ro_entry = _resolve_mcp_entry(ro_script)
    exec_entry = _resolve_mcp_entry(exec_script)
    artifacts_entry = _resolve_mcp_entry(artifacts_script)
    net_entry = _resolve_mcp_entry(net_script)
    shell_entry = _resolve_mcp_entry(shell_script)

    for p in (ro_entry, exec_entry, artifacts_entry, net_entry, shell_entry):
        # For module mode (module:xxx), skip file existence checks.
        if isinstance(p, str) and p.startswith("module:"):
            continue
        if not Path(p).exists():
            raise FileNotFoundError(f"MCP server script not found: {p}")

    # Load tools from MCP servers using temporary=True
    ro_tools = await CLIENT_MANAGER.get_tools(ro_entry, temporary=True)
    exec_tools = await CLIENT_MANAGER.get_tools(exec_entry, temporary=True)
    artifact_tools = await CLIENT_MANAGER.get_tools(artifacts_entry, temporary=True)
    web_tools = await CLIENT_MANAGER.get_tools(net_entry, temporary=True)
    shell_tools = await CLIENT_MANAGER.get_tools(shell_entry, temporary=True)

    # Update global registry for safe_tool_node
    all_tools = ro_tools + exec_tools + artifact_tools + web_tools + shell_tools
    update_global_registry(all_tools)

    return {
        "ro": ro_tools + web_tools + shell_tools,
        "exec": ro_tools + exec_tools + web_tools + shell_tools,
        "artifacts": ro_tools + artifact_tools + web_tools + shell_tools,
        # Specialized
        "researcher": all_tools,
        "solver": all_tools,
        "writer": all_tools,
        "code_researcher": all_tools,
        "code_reviewer": all_tools,
    }


def build_llm():
    intent_base = mk_role_chat("intent")
    planner_base = mk_role_chat("planner")
    reflector_base = mk_role_chat("reflector")

    intent_llm = with_structured(intent_base, IntentModel)
    planner_llm = with_structured(planner_base, PlanModel)
    reflector_llm = with_structured(reflector_base, ReflectModel)

    # Tool-enabled agent LLM (base only, tools bound later)
    agent_base = mk_role_chat("agent")

    # Plain LLM (no tools) used by final responder.
    responder_llm: BaseChatModel = agent_base

    return intent_llm, planner_llm, agent_base, reflector_llm, responder_llm


@utils.timer
async def intent_node(state, intent_llm):
    user_req = state["user_request"]

    # Only keep a short history preview (optional); don't str() the BaseMessage list directly
    history = list(state.get("messages") or [])
    history_preview = msg_preview(history, n=LLM_INTENT_HISTORY_PREVIEW_MSGS)

    msgs = [
        SystemMessage(content=INTENT_SYSTEM),
        utils.runtime_clock_msg(),
        HumanMessage(content=f"用户需求：{user_req}"),
        HumanMessage(content=f"历史预览（仅供参考，可忽略）：{json.dumps(history_preview, ensure_ascii=False)}"),
    ]

    try:
        intent_obj = await async_invoke_structured_with_retry(
            intent_llm, msgs, role="intent", retries=2
        )
        intent_dict = to_dict(intent_obj)
        intent_dict = normalize_intent_dict(intent_dict)

        usage = intent_obj.get("usage_metadata", {}) if isinstance(intent_obj, dict) else {}
    except Exception as e:
        log_event(logging.ERROR, "intent.parse_fallback", err=str(e), user_request=user_req)
        intent_dict = {
            "task_type": "other",
            "user_goal": user_req[:200],
            "domains": [],
            "deliverable": "direct_answer",
            "entities": [],
            "need_web": False,
            "suggested_tools": [],
            "constraints": [],
            "missing_info": [],
            "output_language": "zh",
            "role_preset": {},
        }
        usage = {}

    return {
        "intent": intent_dict,
        "usage_metadata": usage,
    }


@utils.timer
async def initial_plan_node(state: AgentState, llm) -> Dict[str, Any]:
    user_req = state["user_request"]

    log_event(
        logging.INFO,
        "initial_plan.start",
        user_request=user_req,
        seen_steps=len(state.get("seen_step_hashes", []) or []),
    )

    intent_data = state.get("intent") or {}
    intent_json = json.dumps(intent_data, ensure_ascii=False)

    # Check for plan_review feedback
    review_feedback = state.get("plan_review_feedback")
    feedback_msg = ""
    if review_feedback:
        feedback_msg = f"\n\n【重要】上一版计划被驳回。修改意见：\n{review_feedback}\n请务必根据意见修正计划。"

    # Record role presets for debugging/verification
    role_preset = intent_data.get("role_preset", {})

    log_event(
        logging.INFO,
        "initial_plan.role_preset",
        preset_keys=list(role_preset.keys()),
        preset_summary={
            k: (v[:100] + "..." if len(v) > 100 else v) for k, v in role_preset.items()
        },
    )

    msgs = [
        SystemMessage(content=INTENT_SYSTEM),
        utils.runtime_clock_msg(),
        SystemMessage(content=f"已完成意图分析（JSON）：{intent_json}"),
        HumanMessage(content=f"用户需求：{user_req}{feedback_msg}\n请生成前2-3个关键步骤，后续步骤将在执行中动态生成。"),
    ]

    t0 = time.time()
    try:
        plan_model = await async_invoke_structured_with_retry(
            llm, msgs, role="plan", retries=3
        )
        plan_obj = to_dict(plan_model)
        ok = True

        # Extract token usage
        plan_usage = plan_model.get("usage_metadata", {}) if isinstance(plan_model, dict) else {}
    except Exception as e:
        ok = False
        log_event(logging.ERROR, "initial_plan.error", err=f"{type(e).__name__}: {e}")
        plan_obj = {
            "version": 1,
            "objective": user_req,
            "steps": [
                {
                    "id": "step1",
                    "title": "初步分析",
                    "agent": "researcher",
                    "task": f"初步分析用户需求：{user_req}",
                    "acceptance": "输出初步分析结果",
                    "inputs": {},
                    "depends_on": [],
                }
            ],
        }
        plan_usage = {}

    # Dedup: skip steps already completed (has artifact)
    seen = set(state.get("seen_step_hashes", []))
    dedup_steps = []
    for st in plan_obj.get("steps", []):
        st_dict = st.model_dump() if hasattr(st, "model_dump") else st
        h = hash_step(st_dict)
        if h in seen:
            continue
        # Don't add hash to seen until the step is completed (accepted by reflect)
        dedup_steps.append(st_dict)

    plan_obj["steps"] = dedup_steps

    steps = plan_obj.get("steps") or []
    if not steps:
        plan_obj = {
            "version": 1,
            "objective": user_req,
            "steps": [
                {
                    "id": "step1",
                    "title": "初步分析",
                    "agent": "researcher",
                    "task": f"初步分析用户需求：{user_req}",
                    "acceptance": "输出初步分析结果",
                    "inputs": {},
                    "depends_on": [],
                }
            ],
        }

    log_event(
        logging.INFO,
        "initial_plan.done",
        ok=ok,
        cost_ms=int((time.time() - t0) * 1000),
        steps=len(plan_obj.get("steps") or []),
        step_brief=[
            f'{s.get("id")}:{s.get("agent")}' for s in (plan_obj.get("steps") or [])
        ][:12],
    )

    return {
        "plan": plan_obj,
        "step_idx": 0,
        "executed_steps": [],
        "pending_steps": [],
        "iter_count": 0,
        "max_iters": state.get("max_iters", MAX_ITERATIONS),
        "done": False,
        "seen_step_hashes": list(seen),
        # === Fix: clear stats to avoid data pollution when Replan reuses step IDs ===
        "step_tool_stats": {},
        "step_failures": {},
        "no_progress": {},
        "last_feedback": {},
        "plan_review_feedback": None,  # Clear old review feedback
        "usage_metadata": plan_usage,  # Add token usage
    }


@utils.timer
async def dynamic_plan_node(state: AgentState, llm) -> Dict[str, Any]:
    user_req = state["user_request"]

    log_event(
        logging.INFO,
        "dynamic_plan.start",
        user_request=user_req,
        seen_steps=len(state.get("seen_step_hashes", []) or []),
    )

    intent_data = state.get("intent") or {}

    # Record role presets for debugging/verification
    role_preset = intent_data.get("role_preset", {})

    log_event(
        logging.INFO,
        "dynamic_plan.role_preset",
        preset_keys=list(role_preset.keys()),
        preset_summary={
            k: (v[:100] + "..." if len(v) > 100 else v) for k, v in role_preset.items()
        },
    )

    review_feedback = state.get("plan_review_feedback")
    feedback_msg = ""
    if review_feedback:
        feedback_msg = f"\n\n【重要】上一版计划被驳回。修改意见：\n{review_feedback}\n请务必根据意见修正计划。"

    executed_steps = state.get("executed_steps", [])
    artifacts = state.get("artifacts", {})

    # Get recent execution results
    recent_artifacts = []
    for step_id in executed_steps[-3:]:  # Results of the last 3 steps
        if step_id in artifacts:
            recent_artifacts.append({
                "step_id": step_id,
                "result": artifacts[step_id].get("content", "")[:500]  # Truncate to avoid being too long
            })

    msgs = [
        SystemMessage(content=PLAN_SYSTEM),
        utils.runtime_clock_msg(),
        SystemMessage(content=f"用户目标：{user_req}{feedback_msg}"),
        SystemMessage(content=f"已完成步骤：{executed_steps}"),
        SystemMessage(content=f"最近执行结果：{json.dumps(recent_artifacts, ensure_ascii=False)}"),
        HumanMessage(content="请生成下一个需要执行的步骤。如果任务已经完成，请返回空步骤。")
    ]

    t0 = time.time()
    try:
        plan_model = await async_invoke_structured_with_retry(
            llm, msgs, role="plan", retries=3
        )
        plan_obj = to_dict(plan_model)
        ok = True
        # Extract token usage
        plan_usage = plan_model.get("usage_metadata", {}) if isinstance(plan_model, dict) else {}
        # Extract new step(s)
        new_steps = plan_obj.get("steps", [])
    except Exception as e:
        ok = False
        log_event(logging.ERROR, "dynamic_plan.error", err=f"{type(e).__name__}: {e}")

        # Generate a default step
        plan_obj = {
            "version": 1,
            "objective": user_req,
            "steps": [{
                "id": f"dynamic_step_{int(time.time())}",
                "title": "继续处理",
                "agent": "writer",
                "task": "继续处理用户请求",
                "acceptance": "输出处理结果",
                "inputs": {},
                "depends_on": [],
            }],
        }
        new_steps = plan_obj.get("steps", [])
        plan_usage = {}

    # === Fix: append new steps to the existing plan to keep step_idx continuity ===
    existing_plan = state.get("plan") or {}
    existing_steps = existing_plan.get("steps") or []

    # Preserve the original plan objective when possible
    if existing_plan.get("objective"):
        plan_obj["objective"] = existing_plan["objective"]

    # Dedup: ensure newly generated steps don't duplicate existing ones (by hash)
    seen = set(state.get("seen_step_hashes", []))
    for s in existing_steps:
        s_dict = s.model_dump() if hasattr(s, "model_dump") else s
        seen.add(hash_step(s_dict))

    dedup_new_steps = []
    for st in new_steps:
        st_dict = st.model_dump() if hasattr(st, "model_dump") else st
        h = hash_step(st_dict)
        if h in seen:
            continue
        dedup_new_steps.append(st_dict)

    if not dedup_new_steps:
        # If dedup results in no new steps but we still need one, fall back to a default step
        dedup_new_steps = [{
            "id": f"dynamic_step_{int(time.time())}",
            "title": "继续处理",
            "agent": "writer",
            "task": "继续处理用户请求",
            "acceptance": "输出处理结果",
            "inputs": {},
            "depends_on": [],
        }]

    # Merge plans
    final_steps = existing_steps + dedup_new_steps
    plan_obj["steps"] = final_steps

    # Keep pending_steps for debugging
    pending_steps = state.get("pending_steps", [])
    pending_steps.extend(dedup_new_steps)

    log_event(
        logging.INFO,
        "plan.done",
        ok=ok,
        cost_ms=int((time.time() - t0) * 1000),
        new_steps=len(dedup_new_steps),
        total_steps=len(final_steps),
        step_brief=[
            f'{s.get("id")}:{s.get("agent")}' for s in dedup_new_steps
        ][:12],
    )

    return {
        "plan": plan_obj,
        "executed_steps": executed_steps,
        "pending_steps": pending_steps,
        "need_plan_update": False,
        "iter_count": 0,
        "max_iters": state.get("max_iters", MAX_ITERATIONS),
        "done": False,
        "seen_step_hashes": list(seen),
        # === Fix: clear stats to avoid data pollution when Replan reuses step IDs ===
        "step_tool_stats": {},
        "step_failures": {},
        "no_progress": {},
        "last_feedback": {},
        "plan_review_feedback": None,  # Clear old review feedback
        "usage_metadata": plan_usage,  # Add token usage
    }


@utils.timer
async def plan_reviewer_node(state: AgentState, llm) -> Dict[str, Any]:
    plan = state.get("plan")
    user_req = state.get("user_request")

    msgs = [
        SystemMessage(content=PLAN_REVIEW_SYSTEM),
        HumanMessage(content=f"用户需求：{user_req}\n\n待审核计划（JSON）：{json.dumps(plan, ensure_ascii=False)}")
    ]

    try:
        review_model = await async_invoke_structured_with_retry(
            llm, msgs, role="plan_reviewer", retries=2
        )
        review_obj = to_dict(review_model)
    except Exception as e:
        log_event(logging.ERROR, "plan_review.error", err=f"{type(e).__name__}: {e}")
        # If review fails, default to approve to avoid blocking the flow
        review_obj = {"decision": "approve", "feedback": ""}

    decision = review_obj.get("decision", "approve")
    feedback = review_obj.get("feedback", "")

    log_event(
        logging.INFO,
        "plan_review.done",
        decision=decision,
        feedback=truncate(feedback, 200)
    )

    updates = {
        "plan_review_feedback": feedback if decision == "reject" else None,
    }

    if decision == "reject":
        # Reject logic: return to the originating planning node
        # We need to know if we came from initial_plan or dynamic_plan.
        # However, in the current new design, Reviewer ONLY serves Dynamic Plan.
        # But to be safe and generic, we can check if executed_steps is empty.

        # Or simpler: The user requested Reviewer ONLY for Dynamic Plan.
        # So we default rejection to "dynamic_plan".
        # (If we ever reconnect Initial Plan -> Reviewer, we'd need a flag in state like "last_plan_source")
        updates["next_node"] = "dynamic_plan"
    else:
        updates["next_node"] = "route"

    return updates


# ===== Skills injection (B + optional C) =====
# B: System-side retrieval of top-k skill cards relevant to the current task and compatible with the role toolset.
# C: Optional LLM selection (router) to choose a small subset of skills and inline their full docs/excerpts.

SKILL_CARD_TOPK = int(os.getenv("SKILL_CARD_TOPK", "8") or 8)
SKILL_SELECT_TOPK = int(os.getenv("SKILL_SELECT_TOPK", "2") or 2)  # keep small to control token cost
SKILL_CONTEXT_MAX_CHARS = int(os.getenv("SKILL_CONTEXT_MAX_CHARS", "12000") or 12000)
SKILL_SELECT_MODE = str(os.getenv("SKILL_SELECT_MODE", "b+c") or "b+c").lower()


# Modes:
# - "off": disable skills injection
# - "b": inject only skill cards (summary)
# - "b+c": inject cards and use an LLM router to inline a few full skill docs
# - "heuristic": same as "b" but deterministically pick top-N skills without LLM routing
# Default is "b+c".
class SkillSelectModel(BaseModel):
    selected_skill_ids: List[str] = Field(default_factory=list,
                                          description="Select up to SKILL_SELECT_TOPK skill ids from the provided candidates.")


_SKILL_SELECT_LLM = None


def _get_skill_select_llm():
    global _SKILL_SELECT_LLM
    if _SKILL_SELECT_LLM is None:
        # Use a tool-free model; structured output ensures we get a parseable list.
        _SKILL_SELECT_LLM = with_structured(mk_role_chat("planner"), SkillSelectModel)
    return _SKILL_SELECT_LLM


def _tool_names(role_tools: Optional[List[Any]]) -> List[str]:
    if not role_tools:
        return []
    names: List[str] = []
    for t in role_tools:
        n = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
        n = str(n).strip()
        if n:
            names.append(n)
    return names


async def _build_skill_injection(role: str, role_tools: Optional[List[Any]], query_text: str) -> str:
    if SKILL_SELECT_MODE in ("off", "0", "false", "none"):
        return ""
    toolset = set(_tool_names(role_tools))
    if not toolset:
        return ""
    cards = SKILL_REGISTRY.search_cards(query_text or "", toolset=toolset, top_k=SKILL_CARD_TOPK)
    if not cards:
        return ""
    cards_text = SKILL_REGISTRY.render_cards(cards, max_items=SKILL_CARD_TOPK)

    mode = SKILL_SELECT_MODE
    if mode in ("b", "cards"):
        return cards_text

    # Deterministic top-N selection (no extra LLM call)
    if mode in ("heuristic", "topn"):
        selected = [m.skill_id for m in cards[:max(1, SKILL_SELECT_TOPK)]]
        docs_text = SKILL_REGISTRY.render_full_docs(selected, max_chars_total=SKILL_CONTEXT_MAX_CHARS)
        return cards_text + "\n\n" + docs_text if docs_text else cards_text

    # LLM routing selection (C)
    if "c" not in mode:
        return cards_text

    selector = _get_skill_select_llm()  # SkillSelectModel already bound
    cand_ids = [m.skill_id for m in cards]
    select_msgs: List[BaseMessage] = [
        SystemMessage(content=(
            "You are a skill router.\n"
            f"Select up to {SKILL_SELECT_TOPK} skill ids that are most relevant to the user's request + current step.\n"
            "Rules:\n"
            "- Only choose from the provided candidate ids.\n"
            "- Prefer fewer skills; choose none if none are clearly relevant.\n"
            "- Do not output anything except the structured fields."
        )),
        HumanMessage(content=(
            f"User/Step Query:\n{query_text}\n\n"
            f"Candidate Skill Cards (ids): {', '.join(cand_ids)}\n\n"
            f"{cards_text}"
        )),
    ]
    ans = await async_invoke_structured_with_retry(
        selector,
        select_msgs,
        role=role,
        retries=2,
        base_sleep=0.3,
    )

    raw = getattr(ans, "selected_skill_ids", None)
    if raw is None and isinstance(ans, dict):
        raw = ans.get("selected_skill_ids")
    selected_ids: List[str] = []
    for x in (raw or []):
        sid = str(x).strip()
        if sid and sid in cand_ids and sid not in selected_ids:
            selected_ids.append(sid)
        if len(selected_ids) >= max(1, SKILL_SELECT_TOPK):
            break

    if not selected_ids:
        return cards_text

    docs_text = SKILL_REGISTRY.render_full_docs(selected_ids, max_chars_total=SKILL_CONTEXT_MAX_CHARS)
    return cards_text + "\n\n" + docs_text if docs_text else cards_text


# ===== End Skills injection =====

def _tool_allowed_roles() -> set[str]:
    # Inject skills catalog only for roles that execute tasks / call tools
    # planner/router/reflector usually skip injection to save tokens
    return {"solver", "code_researcher", "code_reviewer"}


@utils.timer
def agent_node(role: str, llm: BaseChatModel, role_tools: Optional[List[Any]] = None):
    """
    Step executor (Scheme B). Tools are executed only by ToolNode.
    Token-optimized: only feed step-local messages + required dependency artifacts.
    """

    @utils.timer
    async def _node(state: AgentState) -> Dict[str, Any]:
        step_idx = int(state.get("step_idx", 0) or 0)
        step = current_step(state) or {"id": f"step{step_idx + 1}", "task": "No task", "acceptance": ""}
        step_id = str(step.get("id") or f"step{step_idx + 1}")

        plan = state.get("plan") or {}
        steps = plan.get("steps") or []

        failures = dict(state.get("step_failures") or {})
        fails = int(failures.get(step_id, 0))

        tool_stats = dict(state.get("step_tool_stats") or {})
        current_tool_count = int(tool_stats.get(step_id, 0))

        # Step-local history slice (only this step)
        step_msgs, cursors = step_local_messages(state, step_id)

        # Build minimal prompt
        system_prompt = AGENT_SYSTEMS.get(role, AGENT_SYSTEMS.get("solver", ""))

        cheatsheet = render_tool_cheatsheet(role_tools, max_chars=3500)
        system_prompt = internal_prompt.inject_tool_cheatsheet(
            system_prompt,
            cheatsheet,
        )

        msgs: List[BaseMessage] = [
            SystemMessage(content=system_prompt),
            utils.runtime_clock_msg(),
            SystemMessage(content=TIME_FALLBACK_CONTENT),
        ]

        # Add step-local tool/ai messages (bounded)
        msgs.extend(step_msgs)

        # Feedback for this step only
        fb = (state.get("last_feedback") or {}).get(step_id) or {}
        fb_text = ""
        if fb:
            fb_text = f"\n\n【审阅意见】\n原因：{fb.get('reason', '')}\n要求：{fb.get('required_changes', [])}"

        # If returning from tools, force no more tool calls
        last_msg = step_msgs[-1] if step_msgs else None
        returning_from_tools = isinstance(last_msg, ToolMessage)

        if returning_from_tools:
            user_prompt = (
                f"你是 {role} 角色。\n"
                f"[Step {step_idx + 1}/{max(len(steps), 1)} 状态更新]\n"
                f"检测到工具输出已返回（上方 ToolMessage）。\n"
                f"原任务：{step.get('task', '')}\n"
                f"验收标准：{step.get('acceptance', '')}\n"
                f"最高优先级指令：\n"
                f"1) 禁止再调用任何工具（包括 web_search 等）。\n"
                f"2) 直接基于 ToolMessage 输出本步骤结果。\n"
                f"3) 禁止生成 tool_calls。\n"
                f"{fb_text}"
            )
        else:
            user_prompt = (
                f"你是 {role} 角色。请完成当前 step 的任务。\n"
                f"[Step {step_idx + 1}/{max(len(steps), 1)}]\n"
                f"任务：{step.get('task', '')}\n"
                f"验收标准：{step.get('acceptance', '')}\n"
                f"当前重试次数：{fails}\n"
                f"专注原则：仅执行当前 Task。满足验收标准后立刻停止。\n"
                f"{AGENT_RETRY_INSTRUCTION if fails > 0 else ''}"
                f"{fb_text}"
            )

        # Skills injection: inject role-compatible, query-relevant skill cards (and optionally inline a few full docs).
        if (not returning_from_tools) and (role in _tool_allowed_roles()):
            query_text = "\n".join([x for x in [
                str(state.get("user_request") or ""),
                str(plan.get("objective") or ""),
                str(step.get("task") or ""),
                str(step.get("acceptance") or ""),
            ] if x.strip()])
            skill_ctx = await _build_skill_injection("skill_router", role_tools, query_text)
            if skill_ctx.strip():
                msgs.append(SystemMessage(content=skill_ctx))

        msgs.append(HumanMessage(content=user_prompt))

        try:
            resp = await async_invoke_chat_with_retry(llm, msgs, role=role, retries=2)
        except Exception as e:
            resp = AIMessage(content=f"系统内部错误：LLM 不可用（{type(e).__name__}: {e}）。")

        tool_calls = extract_tool_calls(resp)
        if tool_calls:
            tool_stats[step_id] = current_tool_count + len(tool_calls)
            usage = ai_meta(resp).get("usage_metadata", {})
            return {
                "last_step_id": step_id,
                "last_agent_role": role,
                "step_tool_stats": tool_stats,
                "messages": [resp],
                "step_msg_start": cursors,
                "usage_metadata": usage,
            }

        new_text = extract_text_content(resp).strip()
        tool_sig = tool_calls_signature(tool_calls)

        last_output_hash = dict(state.get("last_output_hash") or {})
        last_hash = last_output_hash.get(step_id, "")
        new_hash = text_hash(new_text + "\n" + tool_sig)
        last_output_hash[step_id] = new_hash

        no_progress = dict(state.get("no_progress") or {})
        no_progress[step_id] = (not new_text and not tool_calls) or (fails > 0 and last_hash == new_hash)

        artifacts = dict(state.get("artifacts") or {})
        artifacts[step_id] = {
            "role": role,
            "content": new_text,
            "acceptance": str(step.get("acceptance", "")),
            "task": str(step.get("task", "")),
            "attempt": fails + 1,
            "tool_calls_count": int(tool_stats.get(step_id, 0)),
        }

        usage = ai_meta(resp).get("usage_metadata", {})
        return {
            "artifacts": artifacts,
            "last_step_id": step_id,
            "last_agent_role": role,
            "last_output_hash": last_output_hash,
            "no_progress": no_progress,
            "messages": [resp],
            "step_msg_start": cursors,
            "usage_metadata": usage,
        }

    return _node


@utils.timer
async def respond_node(state: AgentState, responder_llm: BaseChatModel):
    # Defensive: normalize steps to avoid `.get` crash if steps contain str
    state, plan_fixed, plan_fix_reason = _normalize_plan_in_state(state)
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    artifacts = state.get("artifacts") or {}
    user_request = state.get("user_request", "")
    reflections = state.get("reflections", [])

    results = []
    for i, s in enumerate(steps):
        step_id = str(s.get("id") or f"step{i + 1}")
        a = artifacts.get(step_id, {})
        out = truncate(a.get("content", ""), 1200)
        results.append(
            f"- Step {i + 1} ({step_id}) role={a.get('role', '')} attempt={a.get('attempt', '')}\n"
            f"  task={truncate(s.get('task', ''), 300)}\n"
            f"  output={out}"
        )

    msgs = [
        SystemMessage(content=RESPOND_SYSTEM + "\n\n【硬性约束】禁止调用工具；直接输出最终答复文本。"),
        HumanMessage(content=(
                f"用户原始请求：{user_request}\n\n"
                f"请基于以下步骤产出合并最终答复：\n" + "\n".join(results) +
                (f"\n\n【审阅记录】{json.dumps(reflections, ensure_ascii=False)}" if reflections else "")
        )),
    ]

    resp = await async_invoke_chat_with_retry(responder_llm, msgs, role="respond", retries=2)
    answer = extract_text_content(resp).strip()

    return {
        "final_answer": answer,
    }


@utils.timer
async def route_node(state: AgentState) -> Dict[str, Any]:
    # Defensive: normalize steps to avoid `.get` crash if steps contain str
    state, plan_fixed, plan_fix_reason = _normalize_plan_in_state(state)
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    idx = state.get("step_idx", 0)
    cur = steps[idx] if (0 <= idx < len(steps)) else {}
    iter_count = state.get("iter_count", 0) + 1

    log_event(
        logging.INFO,
        "route.tick",
        iter=iter_count,
        step_idx=idx,
        step_id=cur.get("id"),
        agent=cur.get("agent"),
        plan_fixed=plan_fixed,
        plan_fix_reason=plan_fix_reason,
        max_iters=state.get("max_iters", MAX_ITERATIONS),
        done=state.get("done", False),
    )

    return {
        "iter_count": iter_count,
    }


@utils.timer
async def reflect_node(state: AgentState, llm) -> Dict[str, Any]:
    step = current_step(state) or {}
    step_id = state.get("last_step_id") or step.get("id", "step0")

    artifacts = state.get("artifacts", {}) or {}
    cur_out = dict(artifacts.get(step_id, {}) or {})
    cur_text = (cur_out.get("content") or "").strip()

    failures = dict(state.get("step_failures", {}) or {})
    feedbacks = dict(state.get("last_feedback", {}) or {})
    no_progress = dict(state.get("no_progress", {}) or {})
    fail_cnt = int(failures.get(step_id, 0))

    plan = state.get("plan") or {}
    all_steps = plan.get("steps") or []
    idx = int(state.get("step_idx", 0) or 0)
    is_last_step = (idx >= len(all_steps) - 1) if all_steps else True

    # tool_calls_count from live stats
    tool_stats = dict(state.get("step_tool_stats") or {})
    cur_out["tool_calls_count"] = int(tool_stats.get(step_id, 0))

    # If no artifact, force retry (tool loop / crash)
    if step_id not in artifacts:
        failures[step_id] = fail_cnt + 1
        feedbacks[step_id] = {
            "reason": "未产生有效产物（可能陷入工具循环或异常）。请停止重复工具调用，输出基于已有信息的结果。",
            "required_changes": ["停止重复搜索", "基于已有信息输出", "无法获取则说明局限"],
        }
        # Reset step-local cursor so the next retry won't carry old tool logs
        cursors = dict(state.get("step_msg_start") or {})
        cursors[step_id] = len(state.get("messages") or [])
        return {
            "step_failures": failures,
            "last_feedback": feedbacks,
            "step_msg_start": cursors,
            "next_node": "route",
        }

    previous_reflections = list(state.get("reflections", []) or [])
    msgs = [
        SystemMessage(content=REFLECT_SYSTEM),
        utils.runtime_clock_msg(),
        HumanMessage(content=json.dumps({
            "objective": (state.get("plan") or {}).get("objective", state.get("user_request", "")),
            "step": step,
            "output": cur_out,
            "failures": fail_cnt,
            "no_progress": no_progress.get(step_id, False),
            "is_last_step": is_last_step,
            "previous_reflections": previous_reflections,
        }, ensure_ascii=False)),
    ]

    try:
        obj_wrap = await async_invoke_structured_with_retry(llm, msgs, role="reflect", retries=3)
        obj = to_dict(obj_wrap)
        ok = True
    except Exception as e:
        ok = False
        obj = {
            "decision": "retry",
            "reason": f"reflect failed: {type(e).__name__}: {e}",
            "required_changes": ["必须输出严格 JSON", "不得调用工具"],
        }

    decision = obj.get("decision", "accept")
    reason = obj.get("reason", "")

    reflections = list(state.get("reflections", []) or [])
    reflections.append(obj)

    updates: Dict[str, Any] = {
        "reflections": reflections,
        "no_progress": no_progress,
        "next_node": "route",
        "need_plan_update": False,  # Default: do not trigger dynamic_plan
    }

    # ---- retry ----
    if decision == "retry":
        failures[step_id] = int(failures.get(step_id, 0)) + 1
        feedbacks[step_id] = {
            "reason": reason,
            "required_changes": obj.get("required_changes", []),
        }
        # reset step-local cursor (avoid carrying old tool logs)
        cursors = dict(state.get("step_msg_start") or {})
        cursors[step_id] = len(state.get("messages") or [])
        updates.update({
            "step_failures": failures,
            "last_feedback": feedbacks,
            "step_msg_start": cursors,
        })
        return updates

    # ---- accept ----
    if decision == "accept":
        seen = set(state.get("seen_step_hashes", []) or [])
        seen.add(hash_step(step.model_dump() if hasattr(step, "model_dump") else step))

        executed_steps = list(state.get("executed_steps") or [])
        if step_id not in executed_steps:
            executed_steps.append(step_id)

        updates.update({
            "seen_step_hashes": list(seen),
            "executed_steps": executed_steps,
            "step_idx": idx + 1,
        })

        # ✅If this is the last step and it looks like a wrap-up step, finish directly (avoid meaningless dynamic_plan)
        if is_last_step and looks_like_final_step(step):
            updates.update({"done": True, "next_node": "respond"})
        return updates

    # ---- revise_plan ----
    if decision == "revise_plan":
        updates.update({
            "need_plan_update": True,
            "dynamic_plan_feedback": f"Reflect建议改计划：{reason}；{obj.get('required_changes', [])}",
            "next_node": "dynamic_plan",
        })
        return updates

    # ---- finish ----
    if decision == "finish":
        updates.update({"done": True, "next_node": "respond"})
        return updates

    # ---- generate_next_step ----
    if decision == "generate_next_step":
        nxt = obj.get("next_step")
        if nxt:
            # ✅Directly append a new step to avoid calling planner again in dynamic_plan
            # Defensive: normalize next_step to dict; NEVER append raw str to steps.
            nxt2, changed, _ = _coerce_step_dict(nxt, default_id=f"dynamic_step_{int(time.time())}")

            plan = dict(state.get("plan") or {})
            steps_norm, _fixed, _reason = _coerce_steps_list(plan.get("steps"))
            steps_norm.append(nxt2)
            plan["steps"] = steps_norm

            updates["plan"] = plan
            updates["next_node"] = "route"
            return updates

        # Only trigger dynamic_plan if next_step is not provided
        updates.update({"need_plan_update": True, "next_node": "dynamic_plan"})
        return updates

    # fallback
    updates["step_idx"] = idx + 1
    return updates


@utils.timer
def route_after_plan_review(state: AgentState) -> str:
    nxt = state.get("next_node")
    if nxt == "dynamic_plan":
        return "dynamic_plan"
    if nxt == "initial_plan":
        return "initial_plan"
    return "route"


@utils.timer
def route_after_route(state: AgentState) -> str:
    if state.get("done"):
        return "respond"

    if state.get("iter_count", 0) >= state.get("max_iters", MAX_ITERATIONS):
        return "respond"

    # ✅Only plan dynamically when explicitly needed (safer / fewer tokens / fewer turns)
    if state.get("need_plan_update"):
        return "dynamic_plan"

    # Defensive: normalize steps to avoid `.get` crash if steps contain str
    state, _fixed, _reason = _normalize_plan_in_state(state)
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    idx = int(state.get("step_idx", 0) or 0)

    # ✅No more steps: go to respond directly (finish)
    if idx >= len(steps):
        return "respond"

    cur = steps[idx] if (0 <= idx < len(steps)) else {}
    agent = cur.get("agent", "writer") if isinstance(cur, dict) else "writer"
    return agent


# Router functions
def route_after_intent(state: AgentState) -> str:
    return "initial_plan"


def route_after_initial_plan(state: AgentState) -> str:
    return "route"


def route_after_dynamic_plan(state: AgentState) -> str:
    return "route"


@utils.timer
def route_after_agent(state: AgentState) -> str:
    msgs = state.get("messages") or []

    last = msgs[-1] if msgs else None
    tool_calls = extract_tool_calls(last)

    if tool_calls:
        return "tools"
    return "reflect"


@utils.timer
def route_after_tools(state: AgentState) -> str:
    role = state.get("last_agent_role", "writer")

    # Support all role types; no longer hard-code limits
    valid_roles = ("researcher", "solver", "writer", "code_researcher", "code_reviewer")
    if role not in valid_roles:
        log_event(
            logging.WARNING,
            "route.invalid_role",
            role=role,
            fallback="writer",
            valid_roles=valid_roles
        )
        role = "writer"

    # === Enhanced circuit-breaker logic ===
    messages = state.get("messages", [])

    # Count consecutive ToolMessage in recent history
    tool_consecutive_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], ToolMessage):
            tool_consecutive_count += 1
        elif isinstance(messages[i], AIMessage):
            # If an AI message has tool_calls, it's the initiator in the same turn
            if getattr(messages[i], "tool_calls", None) or messages[
                i
            ].additional_kwargs.get("tool_calls"):
                continue
            else:
                # Plain text reply resets the counter
                break
        elif isinstance(messages[i], HumanMessage):
            break

    # If we keep doing tools n times with no result, we must stop.
    if tool_consecutive_count >= MAX_TOOL_CONSECUTIVE_COUNT:
        log_event(
            logging.WARNING,
            "route.force_reflect",
            reason="too_many_consecutive_tool_calls",
            count=tool_consecutive_count,
        )
        return "reflect"

    return role


@utils.timer
def route_after_reflect(state: AgentState) -> str:
    if state.get("done"):
        return "respond"
    if state.get("iter_count", 0) >= state.get("max_iters", MAX_ITERATIONS):
        return "respond"

    nxt = state.get("next_node")
    if nxt == "dynamic_plan":
        return "dynamic_plan"
    if nxt == "initial_plan":
        return "initial_plan"
    if nxt == "respond":
        return "respond"
    return "route"


def build_reflection_multi_agent_graph(
        intent_base: BaseChatModel,
        planner_llm: BaseChatModel,
        agent_base: BaseChatModel,
        reflector_llm: BaseChatModel,
        responder_llm: Optional[BaseChatModel] = None,
        tool_map: Dict[str, List] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """
    Build the graph.
    tool_map: mapping from role name to list of tools.
    """

    responder_llm = responder_llm or agent_base
    tool_map = tool_map or {}

    # type: ignore
    builder = StateGraph(AgentState)

    # Key goal: ToolNode should not raise on tool errors; return errors as tool outputs to the model
    builder.add_node("tools", safe_tool_node)

    # --- async wrappers: key point! ---
    async def _intent(s: AgentState) -> Dict[str, Any]:
        return await intent_node(s, intent_base)

    async def _initial_plan(s: AgentState) -> Dict[str, Any]:
        return await initial_plan_node(s, planner_llm)

    async def _dynamic_plan(s: AgentState) -> Dict[str, Any]:
        return await dynamic_plan_node(s, planner_llm)

    async def _plan_reviewer(s: AgentState) -> Dict[str, Any]:
        # Reuse planner_llm for review, or use a separate one if needed
        return await plan_reviewer_node(s, planner_llm)

    def _make_agent(role: str):
        # Bind tools specific to the role
        tools = tool_map.get(role, [])
        if tools:
            bound_llm = agent_base.bind_tools(tools, tool_choice="auto")
        else:
            bound_llm = agent_base

        return agent_node(role, bound_llm, tools)

    async def _reflect(s: AgentState) -> Dict[str, Any]:
        return await reflect_node(s, reflector_llm)

    async def _respond(s: AgentState) -> Dict[str, Any]:
        return await respond_node(s, responder_llm)

    # --- register nodes ---
    builder.add_node("intent", _intent)
    builder.add_node("initial_plan", _initial_plan)
    builder.add_node("dynamic_plan", _dynamic_plan)
    builder.add_node("plan_reviewer", _plan_reviewer)
    builder.add_node("route", route_node)

    builder.add_node("researcher", _make_agent("researcher"))
    builder.add_node("solver", _make_agent("solver"))
    builder.add_node("writer", _make_agent("writer"))
    builder.add_node("code_researcher", _make_agent("code_researcher"))
    builder.add_node("code_reviewer", _make_agent("code_reviewer"))

    builder.add_node("reflect", _reflect)
    builder.add_node("respond", _respond)

    # --- edges ---
    builder.add_edge(START, "intent")
    builder.add_edge("intent", "initial_plan")
    builder.add_edge("initial_plan", "route")
    builder.add_edge("dynamic_plan", "plan_reviewer")

    builder.add_conditional_edges(
        "plan_reviewer",
        route_after_plan_review,
        {
            "initial_plan": "initial_plan",
            "dynamic_plan": "dynamic_plan",
            "route": "route",
        },
    )

    builder.add_conditional_edges("route", route_after_route, {
        "researcher": "researcher",
        "solver": "solver",
        "writer": "writer",
        "code_researcher": "code_researcher",
        "code_reviewer": "code_reviewer",
        "dynamic_plan": "dynamic_plan",
        "respond": "respond",
    })

    builder.add_conditional_edges(
        "researcher", route_after_agent, {"tools": "tools", "reflect": "reflect"}
    )
    builder.add_conditional_edges(
        "solver", route_after_agent, {"tools": "tools", "reflect": "reflect"}
    )
    builder.add_conditional_edges(
        "writer", route_after_agent, {"tools": "tools", "reflect": "reflect"}
    )
    builder.add_conditional_edges(
        "code_researcher", route_after_agent, {"tools": "tools", "reflect": "reflect"}
    )
    builder.add_conditional_edges(
        "code_reviewer", route_after_agent, {"tools": "tools", "reflect": "reflect"}
    )

    builder.add_conditional_edges(
        "tools", route_after_tools, {
            "researcher": "researcher",
            "solver": "solver",
            "writer": "writer",
            "code_researcher": "code_researcher",
            "code_reviewer": "code_reviewer",
            "reflect": "reflect",
        }
    )

    builder.add_conditional_edges(
        "reflect", route_after_reflect, {
            "route": "route",
            "initial_plan": "initial_plan",
            "dynamic_plan": "dynamic_plan",
            "plan": "initial_plan",  # fallback
            "respond": "respond",
        }
    )

    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)

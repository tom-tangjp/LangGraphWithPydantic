import hashlib
import json
import logging
import os
import re
import time
from functools import wraps
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Set

import httpx
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage

from src.clawflow.config import get_config, env

logger = logging.getLogger(__name__)

def normalize_provider(p: str) -> str:
    p = (p or "").strip().lower()
    aliases = {
        "qwen": "qwen",
        "dashscope": "qwen",
        "ali": "qwen",
        "doubao": "doubao",
        "ark": "doubao",
        "volc": "doubao",
        "yuanbao": "yuanbao",
        "openai_compat": "openai_compat",
        "openai-compatible": "openai_compat",
        "openai": "openai",
        "ollama": "ollama",
        "deepseek": "deepseek",
        "gemini": "gemini",
        "google": "gemini",
        "genai": "gemini",
        "google_genai": "gemini",
    }
    return aliases.get(p, p)

def check_llm_provider(provider: str, model: str) -> bool:
    lower_provider = normalize_provider(provider)
    key_provider = f"llm.providers.{lower_provider}"
    if get_config(key_provider, default="") == "":
        logger.error(
            f"LLM provider {lower_provider} not configured. Please set {key_provider} in config."
        )
        return False

    # key_model = f"{key_provider}.model"
    # if get_config(key_model, default="") == "":
    #     logger.error(f"LLM provider {lower_provider} not configured. Please set {key_model} in config.")
    #     return False
    # Gemini（Google AI Studio）不依赖 base_url
    if lower_provider != "gemini":
        key_uri = f"{key_provider}.base_url"
        if get_config(key_uri, default="") == "":
            logger.error(
                f"LLM provider {lower_provider} base_url not configured. Please set {key_uri} in config."
            )
            return False
    key_api_key = f"{key_provider}.api_key"
    if lower_provider != "ollama" and get_config(key_api_key, default="") == "":
        logger.error(
            f"LLM provider {lower_provider} api_key not configured. Please set {key_api_key} in config."
        )
        return False
    return True


def to_dumpable(obj):
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def truncate(s: str, n: int = 200) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + "...(truncated)"


def msg_preview(msgs, n=2):
    # Only take content snippets of the last n messages to avoid log explosion
    out = []
    for m in (msgs or [])[-n:]:
        out.append(
            {
                "type": type(m).__name__,
                "content": getattr(m, "content", ""),
            }
        )
    return out


def extract_resp_id(msg: Any) -> str:
    rm = getattr(msg, "response_metadata", None) or {}
    return str(rm.get("id") or rm.get("request_id") or "")


def extract_finish_reason(msg: Any) -> str:
    rm = getattr(msg, "response_metadata", None) or {}
    # OpenAI-compatible providers usually set response_metadata.finish_reason
    fr = rm.get("finish_reason")
    if fr:
        return str(fr)
    # Fallback: some implementations put it in additional_kwargs
    ak = getattr(msg, "additional_kwargs", None) or {}
    fr = ak.get("finish_reason")
    return str(fr or "")


def extract_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """
    Normalize output to: [{id, name, args_raw, args_json?}, ...]
    - LangChain (new): AIMessage.tool_calls = [{"name":..,"args":..,"id":..}, ...]
    - OpenAI raw: additional_kwargs["tool_calls"] = [{"id":..,"function":{"name":..,"arguments":"{...json...}"}}, ...]
    """
    if isinstance(msg, dict) and "raw" in msg:
        msg = msg["raw"]

    out: List[Dict[str, Any]] = []

    # 1) LangChain (new)
    tc = getattr(msg, "tool_calls", None) or []
    if tc:
        for c in tc:
            name = c.get("name") if isinstance(c, dict) else getattr(c, "name", "")
            args = c.get("args") if isinstance(c, dict) else getattr(c, "args", None)
            cid = c.get("id") if isinstance(c, dict) else getattr(c, "id", "")
            item: Dict[str, Any] = {"id": cid, "name": name}
            if args is not None:
                item["args"] = args
            out.append(item)
        return out

    # 2) OpenAI-compatible raw tool_calls (arguments is usually a JSON string)
    ak = getattr(msg, "additional_kwargs", None) or {}
    raw = ak.get("tool_calls") or []
    if raw:
        for c in raw:
            c = c or {}
            fn = (c.get("function") or {}) if isinstance(c, dict) else {}
            name = fn.get("name", "")
            arguments = fn.get("arguments", "")
            item: Dict[str, Any] = {
                "id": c.get("id", ""),
                "name": name,
                "args_raw": arguments,
            }
            # Try to parse arguments as JSON; if it fails, keep raw
            if isinstance(arguments, str) and arguments:
                try:
                    item["args"] = json.loads(arguments)
                except Exception:
                    pass
            elif isinstance(arguments, dict):
                item["args"] = arguments
            out.append(item)
        return out

    # 3) Legacy function_call compatibility (some providers still use this)
    fc = ak.get("function_call")
    if isinstance(fc, dict) and fc.get("name"):
        name = fc.get("name", "")
        arguments = fc.get("arguments", "")
        item: Dict[str, Any] = {"id": "", "name": name, "args_raw": arguments}
        if isinstance(arguments, str) and arguments:
            try:
                item["args"] = json.loads(arguments)
            except Exception:
                pass
        elif isinstance(arguments, dict):
            item["args"] = arguments
        out.append(item)

    return out


def tool_calls_signature(tool_calls: list[Any]) -> str:
    """Stable string signature for tool_calls to support progress detection."""
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(tool_calls)


def extract_text_content(msg: Any) -> str:
    """Extract plain text from a LangChain message (AIMessage/ToolMessage/etc.)."""
    # Structured wrapper returns {"raw": AIMessage, "parsed": ...}
    if isinstance(msg, dict) and "raw" in msg:
        msg = msg["raw"]

    content = getattr(msg, "content", "")
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for it in content:
            if it is None:
                continue
            if isinstance(it, str):
                parts.append(it)
                continue
            if isinstance(it, dict):
                t = it.get("text")
                if t:
                    parts.append(str(t))
                    continue
                for v in it.values():
                    if isinstance(v, str) and v:
                        parts.append(v)
                        break
                continue
            parts.append(str(it))
        return "".join(parts)

    return str(content)


def ai_meta(resp):
    # LangChain AIMessage may carry response_metadata (field names vary by version)
    if resp is None:
        return {}
    meta = getattr(resp, "response_metadata", None) or {}
    usage = getattr(resp, "usage_metadata", None) or {}
    return {"response_metadata": meta, "usage_metadata": usage}


def unwrap_structured(obj: Any) -> Any:
    """
    Support two return shapes:
    1) A Pydantic model (PlanModel/ReflectModel)
    2) A dict wrapper: {"raw":..., "parsed":..., "parsing_error":...}
    """
    if obj is None:
        return None

    # case1: pydantic
    if hasattr(obj, "model_dump"):
        return obj

    # case2: include_raw style
    if isinstance(obj, dict) and "parsed" in obj:
        parsed = obj["parsed"]
        return parsed

    return obj


def to_dict(obj: Any) -> Dict[str, Any]:
    obj = unwrap_structured(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    # Fallback: avoid breaking downstream
    return {"_raw": str(obj)}


def extract_first_json_blob(text: str) -> str | None:
    """Extract the first complete JSON object/array from mixed text (supports escapes and nesting)."""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    # Fast path: already a JSON blob
    if s[0] in "{[":
        try:
            json.loads(s)
            return s
        except Exception:
            pass

    # Strip common code fences
    if s.startswith("```"):
        # Best-effort only; failures are OK (generic extraction below)
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s2 = re.sub(r"\n```$", "", s2)
        s = s2.strip()

    # Generic extraction: find first '{' or '[', then match closing with a stack
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    open_to_close = {"{": "}", "[": "]"}
    close_to_open = {"}": "{", "]": "["}
    stack: list[str] = []
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in open_to_close:
            stack.append(ch)
            continue
        if ch in close_to_open:
            if not stack or stack[-1] != close_to_open[ch]:
                return None
            stack.pop()
            if not stack:
                return s[start : i + 1]

    return None


def try_parse_json(text: str) -> Any | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        blob = extract_first_json_blob(s)
        if not blob:
            return None
        try:
            return json.loads(blob)
        except Exception:
            return None


def tail_messages_tool_safe(
    history: list[BaseMessage],
    max_keep: int = 24,
    extra_keep: int = 8,  # Allow keeping a few extra to complete the tool protocol
) -> list[BaseMessage]:
    """
    Truncate history while ensuring we never emit "ToolMessage without its corresponding AIMessage(tool_calls)".
    Strategy: walk backwards; when seeing ToolMessage, include the whole tool block
    (ToolMessage(s) + preceding AIMessage(tool_calls)).
    If the preceding AIMessage(tool_calls) can't be found, drop the ToolMessage block (avoid invalid sequences).
    Optimization: keep the first SystemMessage (if present) to preserve context.
    """
    if len(history) <= max_keep:
        return history

    max_total = max_keep + max(0, int(extra_keep))
    out: list[BaseMessage] = []
    i = len(history) - 1

    while i >= 0 and len(out) < max_total:
        m = history[i]
        out.append(m)
        i -= 1

        if isinstance(m, ToolMessage):
            # 1) Consume consecutive ToolMessage(s) (one assistant tool_calls may have multiple tool outputs)
            while (
                i >= 0 and isinstance(history[i], ToolMessage) and len(out) < max_total
            ):
                out.append(history[i])
                i -= 1

            # 2) Must include preceding AIMessage(tool_calls)
            if (
                i >= 0
                and isinstance(history[i], AIMessage)
                and getattr(history[i], "tool_calls", None)
            ):
                out.append(history[i])
                i -= 1
            else:
                # No matching tool_calls found: drop collected ToolMessage(s) to avoid 400
                while out and isinstance(out[-1], ToolMessage):
                    out.pop()

    out.reverse()

    # Defensive: if it still starts with ToolMessage, drop leading ToolMessage(s)
    while out and isinstance(out[0], ToolMessage):
        out.pop(0)

    # If still too long, trim from the front but never start at ToolMessage
    if len(out) > max_total:
        out = out[-max_total:]
        while out and isinstance(out[0], ToolMessage):
            out.pop(0)

    # Optimization: keep the first SystemMessage (if present) to preserve context
    if history and isinstance(history[0], SystemMessage):
        first_system = history[0]
        if first_system not in out:
            # If there is space, insert at the beginning
            if len(out) < max_total:
                out.insert(0, first_system)
            else:
                # Replace the last non-critical message? For simplicity, do not replace for now
                pass

    return out


def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def is_recoverable_error(e: Exception) -> bool:
    """
    Determine whether an error is recoverable (i.e., a retry might succeed).
    Recoverable: network timeouts, disconnects, rate limits, temporary service unavailability, etc.
    Non-recoverable: permanent output format errors, unregistered tools, invalid parameters, etc.
    """
    import urllib.error
    import http.client
    import socket

    # Network-related errors (usually recoverable)
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    if isinstance(e, ConnectionError):
        return True
    if isinstance(e, socket.gaierror):
        # DNS resolution failure is usually a transient network issue
        return True
    if isinstance(e, urllib.error.URLError):
        # DNS failure / connection refused are usually recoverable
        return True
    if isinstance(e, http.client.RemoteDisconnected):
        return True

    # Try importing httpx/httpcore-related exceptions
    try:
        import httpx

        if isinstance(e, httpx.ConnectError):
            return True
        if isinstance(e, httpx.TimeoutException):
            return True
    except ImportError:
        pass

    try:
        import httpcore

        if isinstance(e, httpcore.ConnectError):
            return True
        if isinstance(e, httpcore.ReadError):
            return True
        if isinstance(e, httpcore.WriteError):
            return True
    except ImportError:
        pass

    if isinstance(e, urllib.error.HTTPError):
        # 5xx server errors are recoverable; 4xx client errors are usually not
        status = e.code
        if 500 <= status < 600:
            return True  # Server error; retryable
        elif status == 429:  # Rate limit
            return True
        else:
            return False  # 4xx client error; retry won't help

    # Ollama ResponseError (e.g., model not found)
    try:
        from ollama import ResponseError

        if isinstance(e, ResponseError):
            return True
    except ImportError:
        pass

    # OpenAI-compatible API errors
    error_str = str(e).lower()
    if "rate limit" in error_str or "too many requests" in error_str:
        return True
    if "timeout" in error_str:
        return True
    if "connection" in error_str:
        return True
    if "dns" in error_str:
        return True
    if "nodename" in error_str or "servname" in error_str:
        return True

    # Specific non-recoverable errors
    if isinstance(e, ValueError):
        # Parse failures etc. are usually not retryable
        return False
    if isinstance(e, KeyError):
        return False
    if isinstance(e, AttributeError):
        return False

    # Default: unknown error; conservatively treat as non-recoverable
    return False


def retry_with_backoff(retries=3, base_sleep=0.5):
    """
    Generic async retry decorator with exponential backoff and recoverable-error detection.
    Usage:
        @retry_with_backoff(retries=3, base_sleep=0.5)
        async def some_async_function(...):
            ...
    """
    import asyncio

    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not is_recoverable_error(e) or i == retries:
                        break
                    await asyncio.sleep(base_sleep * (2**i))
            raise last_exc if last_exc else Exception("Retry exhausted")

        return wrapper

    return decorator


def runtime_utc_iso() -> str:
    # Example: 2026-01-28T19:36:14Z
    from datetime import datetime, timezone
    import os

    # Check env var override
    override = os.environ.get("RUNTIME_UTC_OVERRIDE")
    if override:
        # Try to parse override time
        try:
            # Support multiple formats: ISO 8601 (2026-01-28T19:36:14Z) or simple datetime
            if "T" in override:
                dt = datetime.fromisoformat(override.replace("Z", "+00:00"))
            else:
                # Assume YYYY-MM-DD HH:MM:SS format
                dt = datetime.strptime(override, "%Y-%m-%d %H:%M:%S")
            # Ensure timezone is UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to parse RUNTIME_UTC_OVERRIDE '{override}': {e}. Using current time."
            )

    # Default to current UTC time
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def runtime_clock_msg() -> SystemMessage:
    return SystemMessage(
        content=(
            f"【运行时信息】当前 UTC 时间：{runtime_utc_iso()}。\n"
            f"当你需要'今天/当前日期/当前时间'时，必须以此为准，禁止凭训练数据猜测年份。"
        )
    )


def inject_time_context(msgs: List[BaseMessage]) -> List[BaseMessage]:
    """
    Automatically inject time context into the message list.

    Rules:
    1. Check whether time context already exists (SystemMessage contains the marker)
    2. If not, insert right after the first SystemMessage
    3. If no SystemMessage exists, insert at the beginning

    Returns: a new message list (may include injected time context)
    """
    # Time context marker
    time_marker = "【运行时信息】当前 UTC 时间："

    # Check whether time context is already present
    for msg in msgs:
        if isinstance(msg, SystemMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and time_marker in content:
                # Time context already present; return original list
                return msgs

    # Build time message
    time_msg = runtime_clock_msg()

    # Find the first SystemMessage position
    insert_index = -1
    for i, msg in enumerate(msgs):
        if isinstance(msg, SystemMessage):
            insert_index = i + 1
            break

    if insert_index == -1:
        # No SystemMessage; insert at beginning
        return [time_msg] + msgs
    else:
        # Insert after the first SystemMessage
        return msgs[:insert_index] + [time_msg] + msgs[insert_index:]


FILTER_KEYS: Set[str] = {"traces", "tool_traces", "_trace", "_ui"}


def _deep_strip(obj: Any, *, filter_keys: Set[str]) -> Any:
    """Recursively remove filter_keys and optionally keys starting with '_' from dicts."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in filter_keys or str(k).startswith("_"):
                continue
            out[k] = _deep_strip(v, filter_keys=filter_keys)
        return out
    if isinstance(obj, list):
        return [_deep_strip(x, filter_keys=filter_keys) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_strip(x, filter_keys=filter_keys) for x in obj)
    return obj


def filter_messages_for_llm(
    msgs: List[BaseMessage], *, filter_keys: Set[str] = FILTER_KEYS
) -> List[BaseMessage]:
    """
    Clean messages before sending to the LLM:
    - Strip filter_keys from content (if dict/list/tuple)
    - Strip filter_keys from additional_kwargs/response_metadata
    - Preserve message types and other fields (avoid breaking concrete message classes)
    """
    cleaned: List[BaseMessage] = []
    for m in msgs:
        # content: usually str; if structured (dict/list), deep-clean it
        content = m.content
        if not isinstance(content, str):
            content = _deep_strip(content, filter_keys=filter_keys)

        # These fields may carry tool/trace/internal info
        additional_kwargs = _deep_strip(
            getattr(m, "additional_kwargs", {}) or {}, filter_keys=filter_keys
        )
        response_metadata = _deep_strip(
            getattr(m, "response_metadata", {}) or {}, filter_keys=filter_keys
        )

        # Most important: do not reconstruct BaseMessage; preserve original message type
        # Use model_copy(update=...) (Pydantic v2) for safe copying
        mm = m.model_copy(
            update={
                "content": content,
                "additional_kwargs": additional_kwargs,
                "response_metadata": response_metadata,
            }
        )
        cleaned.append(mm)
    # print(cleaned)
    return cleaned

def get_workspace_root() -> Path:
    env_root = os.getenv("WORKSPACE_ROOT")
    if env_root is not None:
        return Path(env_root).expanduser().resolve()
    return Path(getenv("WORKSPACE.ROOT", "../../..")).expanduser().resolve()

def get_skills_dir() -> Path:
    env_skills = os.getenv("SKILLS_DIR")
    if env_skills is not None:
        return Path(env_skills).expanduser().resolve()
    return Path(getenv("SKILLS.DIR", get_workspace_root())).expanduser().resolve()

def timer(func):
    """简单计时装饰器。

    兼容同步/异步函数：
    - 同步函数：直接调用
    - 异步函数：await 后返回

    说明：LangGraph 节点若是 async 函数，必须返回真正的 dict；
    如果用同步 wrapper 包裹 async 函数会导致返回 coroutine，从而触发
    `InvalidUpdateError: Expected dict, got <coroutine ...>`。
    """
    import inspect

    # IMPORTANT:
    # `langchain_core.tools.tool` 会把函数转成 Tool 对象（如 StructuredTool），需要保留 `.invoke()`。
    # 如果这里再用普通 wrapper 包一层，会把 Tool 变回 function，从而丢失 `.invoke()`。
    if hasattr(func, "invoke") and callable(getattr(func, "invoke", None)):
        return func

    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def awrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{func.__name__} 耗时: {end - start:.6f} 秒")
            return result

        return awrapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.6f} 秒")
        return result

    return wrapper

def _redact_headers(h):
    h = dict(h)
    for k in list(h.keys()):
        if k.lower() in ("authorization", "api-key"):
            h[k] = "***REDACTED***"
    return h

async def log_request(request: httpx.Request):
    if os.getenv("AGENT_LOG_LLM_DUMP", "0") != "1":
        return

    print("\n===== HTTP REAL REQUEST START =====")
    print(request.method, request.url)
    print("headers:", json.dumps(_redact_headers(request.headers), indent=2, ensure_ascii=False, default=str))
    body = request.content
    try:
        payload = json.loads(body.decode("utf-8"))
        print("body:", json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print("body(raw):", body[:4000])
    print("\n===== HTTP REAL REQUEST END =====")

async def log_response(response: httpx.Response):
    if os.getenv("AGENT_LOG_LLM_DUMP", "0") != "1":
        return

    print("\n===== HTTP REAL RESPONSE START =====")
    print("status:", response.status_code, "url:", response.request.url)

    # 读取响应体（注意：读取会消费流；这里把内容塞回去避免影响上层）
    content = await response.aread()
    response._content = content  # 经验做法：让后续仍可读取

    limit = -1
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"body(decode_error): {e}")
        print("body(raw_bytes):", content[:limit])
        return

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"body(json_error): {e}")
        # 这里按你的要求：raw 输出 decode 后的内容
        if len(text) > limit:
            print("body(raw_text):", text[:limit] + f"...(truncated, total={len(text)})")
        else:
            print("body(raw_text):", text)
        return

    print("body:", json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    print("\n===== HTTP REAL RESPONSE END =====")

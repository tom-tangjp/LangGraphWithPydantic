import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Set

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, SystemMessage

# Configuration management
_CONFIG = None
_CONFIG_PATHS = [
    "config.yaml",
    "config.yml",
    "config.json",
    ".env",
    os.path.expanduser("./config/langgraph-pydantic/config.yaml"),
]

logger = logging.getLogger(__name__)


def _load_config() -> Dict[str, Any]:
    """Load configuration from file(s) and environment variables."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config = {}

    # 1. Load from config files (first found)
    for path in _CONFIG_PATHS:
        if os.path.isfile(path):
            try:
                if path.endswith((".yaml", ".yml")):
                    try:
                        import yaml

                        with open(path, "r", encoding="utf-8") as f:
                            file_config = yaml.safe_load(f)
                    except ImportError:
                        raise ImportError(
                            "PyYAML not installed. Install with `pip install PyYAML`"
                        )
                elif path.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        file_config = json.load(f)
                elif path == ".env":
                    # Simple .env file support (key=value)
                    file_config = {}
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                if "=" in line:
                                    key, val = line.split("=", 1)
                                    file_config[key.strip()] = val.strip()
                else:
                    continue

                # Merge (later files override earlier ones)
                if isinstance(file_config, dict):
                    _deep_update(config, file_config)
                break  # Use first found config file
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to load config file {path}: {e}"
                )
                continue

    # 2. Environment variables override config file
    # Convert environment variables with prefix AGENT_ to nested dict
    env_config = {}
    for key, val in os.environ.items():
        if key.startswith("AGENT_"):
            # Convert AGENT_LLM_MODEL to {"llm": {"model": val}}
            parts = key.lower().split("_")
            current = env_config
            for part in parts[1:-1]:  # skip AGENT and last part
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = val
    _deep_update(config, env_config)

    _CONFIG = config
    return config


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively merge source into target."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value by dot-separated key (e.g., 'llm.model')."""
    config = _load_config()
    if key is None:
        return config

    parts = key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


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
    }
    return aliases.get(p, p)


def env(name: str, default: str | None = None) -> str | None:
    # 1. Check environment variable (original behavior)
    v = os.environ.get(name)
    if v is not None and str(v).strip() != "":
        return v

    # 2. Try to get from config file with dot notation
    key = name.lower()
    # Keep original name as key (e.g., "OPENAI_API_KEY" -> "openai.api.key")
    # key = key.replace("_", ".")

    config_value = get_config(key)
    if config_value is not None:
        return str(config_value)

    # 3. Fallback to default
    return default


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
    key_uri = f"{key_provider}.base_url"
    if get_config(key_uri, default="") == "":
        logger.error(
            f"LLM provider {lower_provider} base_url not configured. Please set {key_uri} in config."
        )
        return False
    key_api_key = f"{key_provider}.api_key"
    if get_config(key_api_key, default="") == "":
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
    # 只取最后 n 条消息的 content 片段，避免日志爆炸
    out = []
    for m in (msgs or [])[-n:]:
        out.append(
            {
                "type": type(m).__name__,
                "content": truncate(getattr(m, "content", ""), 180),
            }
        )
    return out


def extract_resp_id(msg: Any) -> str:
    rm = getattr(msg, "response_metadata", None) or {}
    return str(rm.get("id") or rm.get("request_id") or "")


def extract_finish_reason(msg: Any) -> str:
    rm = getattr(msg, "response_metadata", None) or {}
    # openai-compatible 通常在 response_metadata.finish_reason
    fr = rm.get("finish_reason")
    if fr:
        return str(fr)
    # 兜底：有些实现会塞在 additional_kwargs
    ak = getattr(msg, "additional_kwargs", None) or {}
    fr = ak.get("finish_reason")
    return str(fr or "")


def extract_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """
    统一输出为：[{id,name,args_raw,args_json?}, ...]
    - LangChain 新版：AIMessage.tool_calls = [{"name":..,"args":..,"id":..}, ...]
    - OpenAI 原始：additional_kwargs["tool_calls"] = [{"id":..,"function":{"name":..,"arguments":"{...json...}"}}, ...]
    """
    if isinstance(msg, dict) and "raw" in msg:
        msg = msg["raw"]

    out: List[Dict[str, Any]] = []

    # 1) LangChain 新版
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

    # 2) OpenAI 兼容的 raw tool_calls（arguments 通常是 JSON 字符串）
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
            # 尝试把 arguments 解析成 json；失败则保留 raw
            if isinstance(arguments, str) and arguments:
                try:
                    item["args"] = json.loads(arguments)
                except Exception:
                    pass
            elif isinstance(arguments, dict):
                item["args"] = arguments
            out.append(item)
        return out

    # 3) 兼容更老的 function_call（可选：一些 provider 仍会用这个字段）
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
    # LangChain 的 AIMessage 可能携带 response_metadata（不同版本字段略有差异）
    if resp is None:
        return {}
    meta = getattr(resp, "response_metadata", None) or {}
    usage = getattr(resp, "usage_metadata", None) or {}
    return {"response_metadata": meta, "usage_metadata": usage}


def unwrap_structured(obj: Any) -> Any:
    """
    兼容两种返回：
    1) 直接返回 Pydantic Model (PlanModel/ReflectModel)
    2) 返回 {"raw":..., "parsed":..., "parsing_error":...}
    """
    if obj is None:
        return None

    # case1: pydantic
    if hasattr(obj, "model_dump"):
        return obj

    # case2: include_raw 风格
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
    # 兜底：尽量别让后续炸
    return {"_raw": str(obj)}


def extract_first_json_blob(text: str) -> str | None:
    """从混杂文本中提取第一个完整的 JSON object/array（支持字符串转义、嵌套）。"""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    # 快速路径：本身就是 JSON
    if s[0] in "{[":
        try:
            json.loads(s)
            return s
        except Exception:
            pass

    # 去掉常见 code fence
    if s.startswith("```"):
        # 仅做 best-effort；失败也没关系，后面还有通用提取
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s2 = re.sub(r"\n```$", "", s2)
        s = s2.strip()

    # 通用提取：找第一个 { 或 [，然后用栈匹配到闭合
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
    extra_keep: int = 8,  # 允许为了补齐 tool 协议多保留几条
) -> list[BaseMessage]:
    """
    截断 history，但保证不会出现“ToolMessage 没有对应的 AIMessage(tool_calls)”。
    策略：从尾部向前取消息，遇到 ToolMessage 就把同一段 tool block（ToolMessage... + 前置 AIMessage(tool_calls)）整体纳入；
    如果找不到前置 AIMessage(tool_calls)，就丢弃这段 ToolMessage（避免发出非法序列）。
    优化：保留第一个系统消息（如果存在）以维持上下文。
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
            # 1) 吞掉连续的 ToolMessage（同一个 assistant tool_calls 可能对应多个 tool 输出）
            while (
                i >= 0 and isinstance(history[i], ToolMessage) and len(out) < max_total
            ):
                out.append(history[i])
                i -= 1

            # 2) 必须补上前置 AIMessage(tool_calls)
            if (
                i >= 0
                and isinstance(history[i], AIMessage)
                and getattr(history[i], "tool_calls", None)
            ):
                out.append(history[i])
                i -= 1
            else:
                # 找不到对应的 tool_calls：丢弃刚才收集的 tool 输出，避免触发 400
                while out and isinstance(out[-1], ToolMessage):
                    out.pop()

    out.reverse()

    # 防御：如果仍然从 ToolMessage 开始，直接丢掉前缀 ToolMessage
    while out and isinstance(out[0], ToolMessage):
        out.pop(0)

    # 如果还超长，只从前面裁，但不要裁到 ToolMessage 开头
    if len(out) > max_total:
        out = out[-max_total:]
        while out and isinstance(out[0], ToolMessage):
            out.pop(0)

    # 优化：保留第一个系统消息（如果存在）以维持上下文
    if history and isinstance(history[0], SystemMessage):
        first_system = history[0]
        if first_system not in out:
            # 如果还有空间，插入到开头
            if len(out) < max_total:
                out.insert(0, first_system)
            else:
                # 替换最后一个非关键消息？这里简单替换最后一个非ToolMessage/AIMessage
                # 为了简单，暂时不替换
                pass

    return out


def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def is_recoverable_error(e: Exception) -> bool:
    """
    判断错误是否可恢复（即重试可能成功）。
    可恢复错误：网络超时、连接断开、速率限制、临时服务不可用等。
    不可恢复错误：模型输出格式永久错误、工具未注册、参数错误等。
    """
    import urllib.error
    import http.client
    import socket

    # 网络相关错误（通常可恢复）
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    if isinstance(e, ConnectionError):
        return True
    if isinstance(e, socket.gaierror):
        # DNS解析失败，通常是临时网络问题
        return True
    if isinstance(e, urllib.error.URLError):
        # DNS解析失败、连接被拒绝等可恢复
        return True
    if isinstance(e, http.client.RemoteDisconnected):
        return True

    # 尝试导入 httpx 和 httpcore 相关异常
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
        # 5xx服务器错误可恢复，4xx客户端错误通常不可恢复
        status = e.code
        if 500 <= status < 600:
            return True  # 服务器错误，可重试
        elif status == 429:  # 速率限制
            return True
        else:
            return False  # 4xx客户端错误，重试无效

    # Ollama ResponseError (e.g., model not found)
    try:
        from ollama import ResponseError

        if isinstance(e, ResponseError):
            return True
    except ImportError:
        pass

    # OpenAI兼容API错误
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

    # 特定不可恢复错误
    if isinstance(e, ValueError):
        # 解析失败等，通常重试无效
        return False
    if isinstance(e, KeyError):
        return False
    if isinstance(e, AttributeError):
        return False

    # 默认情况：未知错误，保守起见认为不可恢复
    return False


def retry_with_backoff(retries=3, base_sleep=0.5):
    """
    通用异步重试装饰器，支持指数退避和可恢复错误检测。
    用法：
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
    # 例：2026-01-28T19:36:14Z
    from datetime import datetime, timezone
    import os

    # 检查环境变量覆盖
    override = os.environ.get("RUNTIME_UTC_OVERRIDE")
    if override:
        # 尝试解析覆盖时间
        try:
            # 支持多种格式：ISO 8601 (2026-01-28T19:36:14Z) 或简单日期时间
            if "T" in override:
                dt = datetime.fromisoformat(override.replace("Z", "+00:00"))
            else:
                # 假设是 YYYY-MM-DD HH:MM:SS 格式
                dt = datetime.strptime(override, "%Y-%m-%d %H:%M:%S")
            # 确保时区为 UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Failed to parse RUNTIME_UTC_OVERRIDE '{override}': {e}. Using current time."
            )

    # 默认返回当前 UTC 时间
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
    自动注入时间上下文到消息列表中。

    规则：
    1. 检查是否已包含时间上下文（通过检查系统消息是否包含时间标识符）
    2. 如果没有，在第一个系统消息之后插入时间消息
    3. 如果没有系统消息，在消息列表开头插入时间消息

    返回：新的消息列表（可能已添加时间上下文）
    """
    # 时间上下文标识符
    time_marker = "【运行时信息】当前 UTC 时间："

    # 检查是否已包含时间上下文
    for msg in msgs:
        if isinstance(msg, SystemMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and time_marker in content:
                # 已经包含时间上下文，直接返回原列表
                return msgs

    # 获取时间消息
    time_msg = runtime_clock_msg()

    # 找到第一个系统消息的位置
    insert_index = -1
    for i, msg in enumerate(msgs):
        if isinstance(msg, SystemMessage):
            insert_index = i + 1
            break

    if insert_index == -1:
        # 没有系统消息，插入到开头
        return [time_msg] + msgs
    else:
        # 在第一个系统消息之后插入
        return msgs[:insert_index] + [time_msg] + msgs[insert_index:]


FILTER_KEYS: Set[str] = {"traces", "tool_traces", "_trace", "_ui"}


def _deep_strip(obj: Any, *, filter_keys: Set[str]) -> Any:
    """递归移除 dict 中的 filter_keys，以及以 '_' 开头的 key（可选）。"""
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
    对即将发给 LLM 的 messages 做清洗：
    - 清理 content（若为 dict/list/tuple）中的 filter_keys
    - 清理 additional_kwargs、response_metadata 中的 filter_keys
    - 保留消息类型与其它字段，避免 BaseMessage/具体消息类被破坏
    """
    cleaned: List[BaseMessage] = []
    for m in msgs:
        # content: 通常是 str；如果是结构化（dict/list），做深清洗
        content = m.content
        if not isinstance(content, str):
            content = _deep_strip(content, filter_keys=filter_keys)

        # 这些字段有时会携带工具/trace/内部信息
        additional_kwargs = _deep_strip(
            getattr(m, "additional_kwargs", {}) or {}, filter_keys=filter_keys
        )
        response_metadata = _deep_strip(
            getattr(m, "response_metadata", {}) or {}, filter_keys=filter_keys
        )

        # 最重要：不要用 BaseMessage(...)，要保留原来的消息类型
        # 使用 model_copy(update=...)（Pydantic v2）安全复制
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

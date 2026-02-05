import json
import logging
import socket
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Any

from langchain_core.tools import tool

from const import (
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_RECENCY_DAYS,
    WEB_MAX_CHARS,
    WEB_TIMEOUT_S,
)
from utils import env

logger = logging.getLogger(__name__)


def _http_get_impl(
    url: str, max_chars: int = WEB_MAX_CHARS, timeout_s: int = WEB_TIMEOUT_S
) -> str:
    """Simple HTTP GET implementation."""
    # 关键目标：工具失败不要抛异常，返回结构化错误给 LLM，让图继续跑
    headers: Dict[str, str] = {}

    if not url or not isinstance(url, str):
        return json.dumps(
            {"ok": False, "error": "invalid_url", "url": url}, ensure_ascii=False
        )

    if not url.startswith(("http://", "https://")):
        return json.dumps(
            {"ok": False, "error": "unsupported_scheme", "url": url}, ensure_ascii=False
        )

    # 给一个相对“像浏览器”的 UA，很多站/公共 searxng 会拦截无 UA/默认 UA
    headers.setdefault(
        "User-Agent",
        env("AGENT_HTTP_USER_AGENT", "Mozilla/5.0 (compatible; agent_flow/1.0)"),
    )

    # 构造 Request 时传入 headers
    req = urllib.request.Request(url, headers=headers)

    logger.info('event=http_get.request url="%s" timeout_s=%d', req.full_url, timeout_s)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()

            # 智能获取编码，默认回退到 utf-8
            charset = resp.headers.get_content_charset("utf-8")

            # 读取并解码
            text = raw.decode(charset, errors="replace")

            # 统一返回结构，这里暂时保持你的逻辑，但建议上层调用者注意
            return json.dumps({"ok": True, "content": text[:max_chars], "status": 200})

    except urllib.error.HTTPError as e:
        # 场景：连上了服务器，但服务器返回 404, 403, 500 等
        logger.warning(
            'event=http_get.http_error url="%s" code=%s reason="%s"',
            url,
            e.code,
            getattr(e, "reason", "Unknown"),
        )

        body_preview = ""
        try:
            # 尝试读取错误页面的内容（有些 API 报错会把原因写在 body 里）
            body_preview = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass

        return json.dumps(
            {
                "ok": False,
                "error_type": "HTTPError",
                "status": e.code,
                "reason": str(getattr(e, "reason", "")),
                "url": url,
                "body_preview": body_preview,
            },
            ensure_ascii=False,
        )

    except urllib.error.URLError as e:
        # 【优化 A】场景：断网、DNS解析失败、连接被拒绝
        # e.reason 可能是字符串，也可能是 socket.gaierror 对象
        reason_str = str(e.reason)
        logger.warning('event=http_get.url_error url="%s" reason="%s"', url, reason_str)

        return json.dumps(
            {
                "ok": False,
                "error_type": "URLError",  # 区分错误类型
                "status": None,
                "reason": reason_str,
                "url": url,
            },
            ensure_ascii=False,
        )

    except (socket.timeout, TimeoutError):
        # 【优化 B】场景：连接超时或读取超时
        logger.warning('event=http_get.timeout url="%s" timeout=%s', url, timeout_s)
        return json.dumps(
            {
                "ok": False,
                "error_type": "Timeout",
                "reason": f"Request timed out after {timeout_s}s",
                "url": url,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        # 兜底捕获其他未知异常（如 SSL 握手失败、内存溢出等）
        logger.exception('event=http_get.exception url="%s" err="%s"', url, str(e))
        return json.dumps(
            {
                "ok": False,
                "error_type": "UnhandledException",
                "reason": str(e),
                "url": url,
            },
            ensure_ascii=False,
        )


def http_json(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] | None = None,
    data: Dict[str, Any] | None = None,
    timeout_s: int = 20,
) -> Any:
    req_headers = {"User-Agent": "agent-flow/1.0"}
    if headers:
        req_headers.update(headers)

    body = None
    if data is not None:
        raw = json.dumps(data).encode("utf-8")
        body = raw
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(
        url, data=body, headers=req_headers, method=method.upper()
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    txt = raw.decode("utf-8", errors="replace")

    logger.debug(f"url: {url}, http response: {txt}")

    return json.loads(txt)


@tool("http_get")
def http_get(url: str, max_chars: int = 12000, timeout_s: int = 15) -> str:
    """Simple HTTP GET. Returns first max_chars characters of the response body."""
    return _http_get_impl(url, max_chars=max_chars, timeout_s=timeout_s)


def map_recency_to_timerange(recency_days: int) -> str | None:
    # SearxNG time_range: day|week|month|year
    if recency_days <= 0:  # <--- 新增：0 或负数代表不限时间
        return None
    if recency_days <= 1:
        return "day"
    if recency_days <= 7:
        return "week"
    if recency_days <= 31:
        return "month"
    if recency_days <= 366:
        return "year"
    return None


class SearchBackend:
    name: str = "base"

    def search(
        self,
        query: str,
        *,
        recency_days: int = WEB_SEARCH_RECENCY_DAYS,
        domains: List[str] | None = None,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class NoneBackend(SearchBackend):
    name = "none"

    def search(
        self,
        query: str,
        *,
        recency_days: int = WEB_SEARCH_RECENCY_DAYS,
        domains: List[str] | None = None,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        return []


class SearxNGBackend(SearchBackend):
    name = "searxng"

    def __init__(self, base_url: str):
        u = (base_url or "").rstrip("/")
        if u.endswith("/search"):
            u = u[: -len("/search")]
        self.base_url = u

    def search(
        self,
        query: str,
        *,
        recency_days: int = WEB_SEARCH_RECENCY_DAYS,
        domains: List[str] | None = None,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        q = query
        if domains:
            dom = " OR ".join([f"site:{d}" for d in domains])
            q = f"({query}) ({dom})"
        params = {"q": q, "format": "json", "language": "auto"}
        tr = map_recency_to_timerange(int(recency_days))
        if tr:
            params["time_range"] = tr
        url = self.base_url + "/search?" + urllib.parse.urlencode(params)
        data = http_json(url, method="GET", timeout_s=20)

        logger.debug(f"url: {url}, response data: {data}")

        results: List[Dict[str, Any]] = []
        for i, it in enumerate((data or {}).get("results", [])[:max_results]):
            results.append(
                {
                    "rank": i + 1,
                    "title": it.get("title") or "",
                    "url": it.get("url") or "",
                    "snippet": it.get("content") or "",
                    "source": "searxng",
                }
            )
        return results


class TavilyBackend(SearchBackend):
    name = "tavily"
    backend_url = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(
        self,
        query: str,
        *,
        recency_days: int = WEB_SEARCH_RECENCY_DAYS,
        domains: List[str] | None = None,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": int(max_results),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
        if domains:
            payload["include_domains"] = domains
        data = http_json(
            TavilyBackend.backend_url, method="POST", data=payload, timeout_s=30
        )

        logger.debug(f"url: {TavilyBackend.backend_url}, response data: {data}")

        results: List[Dict[str, Any]] = []
        for i, it in enumerate((data or {}).get("results", [])[:max_results]):
            results.append(
                {
                    "rank": i + 1,
                    "title": it.get("title") or "",
                    "url": it.get("url") or "",
                    "snippet": it.get("content") or "",
                    "source": "tavily",
                }
            )
        return results


class SerpAPIBackend(SearchBackend):
    name = "serpapi"
    backend_url = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, engine: str = "google"):
        self.api_key = api_key
        self.engine = engine

    def search(
        self,
        query: str,
        *,
        recency_days: int = WEB_SEARCH_RECENCY_DAYS,
        domains: List[str] | None = None,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        q = query
        if domains:
            dom = " OR ".join([f"site:{d}" for d in domains])
            q = f"({query}) ({dom})"
        params = {"engine": self.engine, "q": q, "api_key": self.api_key}
        url = SerpAPIBackend.backend_url + "?" + urllib.parse.urlencode(params)
        data = http_json(url, method="GET", timeout_s=30)

        logger.debug(f"url: {url}, response data: {data}")

        results: List[Dict[str, Any]] = []
        for i, it in enumerate((data or {}).get("organic_results", [])[:max_results]):
            results.append(
                {
                    "rank": i + 1,
                    "title": it.get("title") or "",
                    "url": it.get("link") or "",
                    "snippet": it.get("snippet") or "",
                    "source": "serpapi",
                }
            )
        return results


class BingBackend(SearchBackend):
    name = "bing"
    backend_url = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(
        self,
        query: str,
        *,
        recency_days: int = WEB_SEARCH_RECENCY_DAYS,
        domains: List[str] | None = None,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
    ) -> List[Dict[str, Any]]:
        q = query
        if domains:
            dom = " OR ".join([f"site:{d}" for d in domains])
            q = f"({query}) ({dom})"
        params = {"q": q, "count": int(max_results)}
        url = BingBackend.backend_url + "?" + urllib.parse.urlencode(params)
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        data = http_json(url, method="GET", headers=headers, timeout_s=30)

        logger.debug(f"url: {url}, response data: {data}")

        items = (((data or {}).get("webPages") or {}).get("value") or [])[:max_results]
        results: List[Dict[str, Any]] = []
        for i, it in enumerate(items):
            results.append(
                {
                    "rank": i + 1,
                    "title": it.get("name") or "",
                    "url": it.get("url") or "",
                    "snippet": it.get("snippet") or "",
                    "source": "bing",
                }
            )
        return results


def _select_search_backend() -> SearchBackend:
    backend = (env("SEARCH.BACKEND") or "").strip().lower()
    if backend:
        if backend == "none":
            return NoneBackend()
        if backend == "searxng":
            url = env("SEARXNG.URL")
            return SearxNGBackend(url) if url else NoneBackend()
        if backend == "tavily":
            key = env("TAVILY.API_KEY")
            return TavilyBackend(key) if key else NoneBackend()
        if backend == "serpapi":
            key = env("SERPAPI.API_KEY")
            return SerpAPIBackend(key) if key else NoneBackend()
        if backend == "bing":
            key = env("BING.API_KEY")
            if not key:
                return NoneBackend()
            return BingBackend(key)
        return NoneBackend()

    # auto selection
    if env("SEARXNG.URL"):
        return SearxNGBackend(env("SEARXNG.URL"))
    if env("TAVILY.API_KEY"):
        return TavilyBackend(env("TAVILY.API_KEY"))
    if env("SERPAPI.API_KEY"):
        return SerpAPIBackend(env("SERPAPI.API_KEY"))
    if env("BING.API_KEY"):
        return BingBackend(env("BING.API_KEY"))
    return NoneBackend()


_SEARCH_BACKEND: SearchBackend = _select_search_backend()


@tool("web_search")
def web_search(
    query: str,
    recency_days: int = WEB_SEARCH_RECENCY_DAYS,
    domains: List[str] | None = None,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """Search the web using the configured backend (searxng/tavily/serpapi/bing/none)."""
    try:
        results = _SEARCH_BACKEND.search(
            query,
            recency_days=int(recency_days),
            domains=domains,
            max_results=int(max_results),
        )
        logger.debug(
            f"domains={domains}, recency_days={recency_days}, max_results={max_results}, query={query}, results: {results}"
        )
        # === 修复：处理空结果 ===
        if not results:
            logger.warning(
                f"No results found for recency_days={recency_days}, max_results={max_results}, domains={domains}, query={query}"
            )
            return [
                {
                    "rank": 0,
                    "title": "No Results",
                    "url": "",
                    "snippet": "No search results found for this query. Please try a broader keyword or check your network.",
                    "source": "system",
                }
            ]

        return results  # 如果原函数签名是 List[Dict]，这里保持原样；如果是 str，则 json.dumps(results)
    except Exception as e:
        req_url = getattr(e, "url", "") or ""
        code = getattr(e, "code", None)
        return [
            {
                "rank": 1,
                "title": "[SEARCH_ERROR]",
                "url": req_url,  # 失败请求的真实 URL（如 HTTPError.url）
                "snippet": f"{type(e).__name__}: {e} (status={code}) | query={query}",
                "source": _SEARCH_BACKEND.name,
            }
        ]


@tool("web_open")
def web_open(
    url: str, max_chars: int = WEB_MAX_CHARS, timeout_s: int = WEB_TIMEOUT_S
) -> str:
    """Open a URL and return text (best effort)."""
    return _http_get_impl(url, max_chars=int(max_chars), timeout_s=int(timeout_s))

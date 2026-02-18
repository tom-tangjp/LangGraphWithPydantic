import json
import logging
import os
import socket
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Dict, Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

WEB_SEARCH_MAX_RESULTS = _env_int("WEB_SEARCH_MAX_RESULTS", 5)
WEB_SEARCH_RECENCY_DAYS = _env_int("WEB_SEARCH_RECENCY_DAYS", 7)
WEB_MAX_CHARS = _env_int("WEB_MAX_CHARS", 20000)
WEB_TIMEOUT_S = _env_int("WEB_TIMEOUT_S", 20)

def _http_get_impl(
    url: str, max_chars: int = WEB_MAX_CHARS, timeout_s: int = WEB_TIMEOUT_S
) -> str:
    """Simple HTTP GET implementation."""
    # Key goal: don't raise on tool failures; return structured errors to the LLM so the graph can continue
    headers: Dict[str, str] = {}

    if not url or not isinstance(url, str):
        return json.dumps(
            {"ok": False, "error": "invalid_url", "url": url}, ensure_ascii=False
        )

    if not url.startswith(("http://", "https://")):
        return json.dumps(
            {"ok": False, "error": "unsupported_scheme", "url": url}, ensure_ascii=False
        )

    # Use a browser-like UA; many sites/public searxng instances block missing/default UA
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (compatible; agent_flow/1.0)",
    )

    # Pass headers when constructing the Request
    req = urllib.request.Request(url, headers=headers)

    logger.info('event=http_get.request url="%s" timeout_s=%d', req.full_url, timeout_s)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()

            # Detect charset; fall back to utf-8
            charset = resp.headers.get_content_charset("utf-8")

            # Read and decode
            text = raw.decode(charset, errors="replace")

            # Return a unified structure; keep current logic for now (callers should handle it)
            return json.dumps({"ok": True, "content": text[:max_chars], "status": 200})

    except urllib.error.HTTPError as e:
        # Case: connected but server returns 404/403/500/etc.
        logger.warning(
            'event=http_get.http_error url="%s" code=%s reason="%s"',
            url,
            e.code,
            getattr(e, "reason", "Unknown"),
        )

        body_preview = ""
        try:
            # Try to read error body preview (some APIs include details in body)
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
        # [Optimization A] Case: offline/DNS failure/connection refused
        # e.reason may be a string or a socket.gaierror object
        reason_str = str(e.reason)
        logger.warning('event=http_get.url_error url="%s" reason="%s"', url, reason_str)

        return json.dumps(
            {
                "ok": False,
                "error_type": "URLError",  # Differentiate error type
                "status": None,
                "reason": reason_str,
                "url": url,
            },
            ensure_ascii=False,
        )

    except (socket.timeout, TimeoutError):
        # [Optimization B] Case: connect timeout or read timeout
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
        # Fallback: catch other unknown exceptions (SSL handshake failure, OOM, etc.)
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
    if recency_days <= 0:  # <--- New: 0 or negative means no time limit
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
    backend = os.getenv("SEARCH_BACKEND", "").strip().lower()
    if backend:
        if backend == "none":
            return NoneBackend()
        if backend == "searxng":
            url = os.getenv("SEARXNG_URL", "").strip()
            return SearxNGBackend(url) if url else NoneBackend()
        if backend == "tavily":
            key = os.getenv("TAVILY_API_KEY", "").strip()
            return TavilyBackend(key) if key else NoneBackend()
        if backend == "serpapi":
            key = os.getenv("SERPAPI_API_KEY", "").strip()
            return SerpAPIBackend(key) if key else NoneBackend()
        if backend == "bing":
            key = os.getenv("BING_API_KEY", "").strip()
            if not key:
                return NoneBackend()
            return BingBackend(key)
        return NoneBackend()

    # auto selection
    searxng_url = os.getenv("SEARXNG_URL", "").strip()
    if searxng_url:
        return SearxNGBackend(searxng_url)
    tavily_url = os.getenv("TAVILY_API_KEY", "").strip()
    if tavily_url:
        return TavilyBackend(tavily_url)
    serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if serpapi_key:
        return SerpAPIBackend(serpapi_key)
    bing_key = os.getenv("BING_API_KEY", "").strip()
    if bing_key:
        return BingBackend(bing_key)
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
        # === Fix: handle empty results ===
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

        return results  # Keep return type consistent with original signature
    except Exception as e:
        req_url = getattr(e, "url", "") or ""
        code = getattr(e, "code", None)
        return [
            {
                "rank": 1,
                "title": "[SEARCH_ERROR]",
                "url": req_url,  # Actual URL of the failed request (e.g. HTTPError.url)
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

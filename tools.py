import json
import logging
import os
import re
import shutil
import subprocess
import ast

from pathlib import Path
from typing import Dict, List, Any

from langchain_core.tools import tool

import utils
from web_tools import web_search, web_open, http_get

logger = logging.getLogger(__name__)


def _resolve_under_root(path: str) -> Path:
    """
    解析路径为相对于 WORKSPACE_ROOT 的绝对路径。
    """
    workspace_root = utils.get_workspace_root()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = workspace_root / p

    rp = p.resolve()

    # 必须位于 WORKSPACE_ROOT 内
    try:
        rp.relative_to(workspace_root)
    except ValueError:
        raise ValueError(f"path escapes workspace root: {rp} (root={workspace_root})")

    return rp


def _get_search_root(path: str) -> Path:
    """
    获取搜索根目录。
    - 如果 path 为空或 "."，使用 WORKSPACE_ROOT（config.yaml 配置）
    - 其他相对路径相对于 WORKSPACE_ROOT
    - 绝对路径直接使用
    """
    if not path or path == ".":
        return utils.get_workspace_root()
    return _resolve_under_root(path)

@utils.timer
@tool("read_file")
def read_file(path: str, start: int = 0, max_chars: int = 12000) -> str:
    """Read a UTF-8 file under WORKSPACE_ROOT. Returns a substring [start:start+max_chars]."""
    fp = _resolve_under_root(path)
    data = fp.read_text(encoding="utf-8", errors="replace")
    if start < 0:
        start = 0
    return data[start: start + max_chars]

@utils.timer
@tool("list_dir")
def list_dir(path: str = ".", max_entries: int = 200) -> List[str]:
    """List directory entries under WORKSPACE_ROOT."""
    dp = _resolve_under_root(path)
    out = []
    for i, child in enumerate(sorted(dp.iterdir(), key=lambda x: x.name)):
        if i >= max_entries:
            break
        out.append(child.name + ("/" if child.is_dir() else ""))
    return out

@utils.timer
@tool("ensure_dir")
def ensure_dir(path: str) -> str:
    """Create a directory under WORKSPACE_ROOT if not exists."""
    dp = _resolve_under_root(path)
    try:
        dp.mkdir(parents=True, exist_ok=True)
        return json.dumps(
            {"ok": True, "path": str(dp.relative_to(utils.get_workspace_root()))},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"ok": False, "error": type(e).__name__, "reason": str(e), "path": path},
            ensure_ascii=False,
        )

@utils.timer
@tool("write_text_file")
def write_text_file(
        path: str, content: str, mode: str = "overwrite", encoding: str = "utf-8"
) -> str:
    """Write UTF-8 text file under WORKSPACE_ROOT. mode=overwrite|append. Uses atomic replace for overwrite."""
    fp = _resolve_under_root(path)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        if mode not in ("overwrite", "append"):
            mode = "overwrite"
        if mode == "append":
            with fp.open("a", encoding=encoding, errors="replace") as f:
                f.write(content or "")
            return json.dumps(
                {
                    "ok": True,
                    "path": str(fp.relative_to(utils.get_workspace_root())),
                    "bytes": len((content or "").encode(encoding, errors="replace")),
                    "mode": "append",
                },
                ensure_ascii=False,
            )
        # overwrite: atomic write then replace
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        with tmp.open("w", encoding=encoding, errors="replace") as f:
            f.write(content or "")
        shutil.move(str(tmp), str(fp))
        return json.dumps(
            {
                "ok": True,
                "path": str(fp.relative_to(utils.get_workspace_root())),
                "bytes": len((content or "").encode(encoding, errors="replace")),
                "mode": "overwrite",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"ok": False, "error": type(e).__name__, "reason": str(e), "path": path},
            ensure_ascii=False,
        )

@utils.timer
@tool("grep_text")
def grep_text(
        pattern: str, path: str = ".", max_matches: int = 50, max_file_size_kb: int = 512
) -> List[Dict[str, Any]]:
    """Search for a regex pattern in text files.

    Args:
        pattern: 正则表达式模式
        path: 搜索目录（"." 表示当前工作目录）
        max_matches: 最大匹配数，默认 50
        max_file_size_kb: 最大文件大小 KB，默认 512
    """
    # 使用正确的搜索根目录
    root = _get_search_root(path)
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from e

    matches: List[Dict[str, Any]] = []
    max_bytes = int(max_file_size_kb) * 1024
    for fp in root.rglob("*"):
        if len(matches) >= int(max_matches):
            break
        if not fp.is_file():
            continue
        try:
            if fp.stat().st_size > max_bytes:
                continue
            data = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for i, line in enumerate(data.splitlines(), start=1):
            if rx.search(line):
                # 计算相对路径
                try:
                    rel_path = str(fp.relative_to(utils.get_workspace_root()))
                except ValueError:
                    rel_path = str(fp)
                matches.append(
                    {
                        "file": rel_path,
                        "line": i,
                        "text": line.strip()[:300],
                    }
                )
                if len(matches) >= int(max_matches):
                    break
    return matches


def build_tool_registry(tools: List[Any]) -> Dict[str, Any]:
    """Build a resilient name->tool map across minor LangChain/LCEL variants."""
    reg: Dict[str, Any] = {}
    for t in tools:
        name = getattr(t, "name", None)
        if isinstance(name, str) and name:
            reg[name] = t
            reg[name.strip()] = t
            reg[name.strip().lower()] = t

        # Some tool wrappers keep the original function name here.
        fn = getattr(t, "func", None)
        fn_name = getattr(fn, "__name__", None)
        if isinstance(fn_name, str) and fn_name:
            reg[fn_name] = t
            reg[fn_name.strip()] = t
            reg[fn_name.strip().lower()] = t
    return reg


# ============================================================================
# 代码分析、编写、测试工具
# ============================================================================

# 允许的安全命令列表（仅限代码分析、测试、格式化等开发任务）
_SAFE_COMMANDS = {
    # python：仅允许少数只读/语法检查用法；更细粒度校验见 _is_safe_python_cmd
    "python": ["-m", "py_compile", "-V", "--version"],
    "python3": ["-m", "py_compile", "-V", "--version"],
    # 测试/静态检查/格式化：允许任意参数（由对应工具/CI兜底）
    "pytest": [],
    "black": [],
    "flake8": [],
    "isort": [],
    # pip：默认只读（show/list/freeze）；写入类动作由 MCP_ALLOW_PIP_WRITE gate
    "pip": ["show", "list", "freeze", "install", "uninstall"],
}

_ALLOW_PIP_WRITE = os.environ.get("MCP_ALLOW_PIP_WRITE", "0") == "1"
_ALLOW_PYTHON_C = os.environ.get("MCP_ALLOW_PYTHON_C", "0") == "1"

# 允许的 python -c one-liner 白名单（可按需扩展；默认关闭）
_PYTHON_C_ALLOWLIST = {
}

def _is_safe_python_cmd(cmd_parts: List[str]) -> bool:
    """
    更严格的 python 校验：
    - 允许: python --version / -V
    - 允许: python -m py_compile <file1> [file2...]
    - 允许: python -m pytest ...
    - 可选: python -c <one-liner>（需 MCP_ALLOW_PYTHON_C=1 且命中白名单）
    其他全部拒绝。
    """
    if not cmd_parts:
        return False
    # python --version / -V
    if len(cmd_parts) == 2 and cmd_parts[1] in ("--version", "-V"):
        return True

    # python -c "<one-liner>" (optional)
    if len(cmd_parts) >= 3 and cmd_parts[1] == "-c":
        if not _ALLOW_PYTHON_C:
            return False

        if len(_PYTHON_C_ALLOWLIST) == 0:
            return True

        one_liner = cmd_parts[2]
        return one_liner in _PYTHON_C_ALLOWLIST

    # python [interp_flags...] -m (py_compile|pytest) ...
    i = 1
    while i < len(cmd_parts) and cmd_parts[i].startswith("-") and cmd_parts[i] not in ("-m", "-c"):
        i += 1
    if i + 1 >= len(cmd_parts) or cmd_parts[i] != "-m":
        return False
    module = cmd_parts[i + 1]
    if module not in ("py_compile", "pytest"):
        return False
    # python -m pytest ... : treat as safe (equivalent to `pytest`)
    if module == "pytest":
        return True
    # python -m py_compile ...
    j = i + 2
    has_file = False
    while j < len(cmd_parts):
        a = cmd_parts[j]
        if a.startswith("-"):
            j += 1
            continue
        try:
            _resolve_under_root(a)
        except Exception:
            return False
        has_file = True
        j += 1
    return has_file

def _is_safe_pip_cmd(cmd_parts: List[str]) -> bool:
    """
    pip 安全校验：
    - 默认允许只读: pip --version/-V, pip show/list/freeze
    - 写入类: pip install/uninstall 仅在 MCP_ALLOW_PIP_WRITE=1 时允许
    """
    if not cmd_parts:
        return False
    if len(cmd_parts) == 2 and cmd_parts[1] in ("--version", "-V"):
        return True
    if len(cmd_parts) < 2:
        return False
    sub = cmd_parts[1]
    readonly = {"show", "list", "freeze"}
    write_ops = {"install", "uninstall"}
    if sub in readonly:
        return True
    if sub in write_ops:
        return _ALLOW_PIP_WRITE
    return False

_SAFE_PYTHON_MODULES = {"py_compile"}  # 只允许这些 -m 模块
_SAFE_PYTHON_FLAGS = {"-V", "--version"}

# 如果你确实想要 python -c，一定要做白名单，而不是放开任意代码
_SAFE_PYTHON_C_ALLOWLIST = {
    "import sys; print(sys.version)",
    "import sys; print(sys.executable)",
}

def _is_under_workspace(path_str: str) -> bool:
    try:
        _resolve_under_root(path_str)  # 会校验不逃逸 WORKSPACE_ROOT
        return True
    except Exception:
        return False

def _is_safe_python_command(cmd_parts: List[str]) -> bool:
    # 1) python --version / -V
    if len(cmd_parts) == 2 and cmd_parts[1] in _SAFE_PYTHON_FLAGS:
        return True

    # 2) python -c "<one-liner>"（严格白名单）
    if len(cmd_parts) == 3 and cmd_parts[1] == "-c" and cmd_parts[2] in _SAFE_PYTHON_C_ALLOWLIST:
        return True

    # 3) python -m py_compile <file...>（允许多个文件；文件必须在 workspace 内）
    if len(cmd_parts) >= 4 and cmd_parts[1] == "-m" and cmd_parts[2] in _SAFE_PYTHON_MODULES:
        for arg in cmd_parts[3:]:
            if arg.startswith("-"):
                continue  # 例如 py_compile 的 -q
            if not _is_under_workspace(arg):
                return False
        return True

    return False

def _is_safe_command(cmd_parts: List[str]) -> bool:
    """检查命令是否在安全列表中（更严格的 python/pip 规则）。"""
    if not cmd_parts:
        return False
    base_cmd = cmd_parts[0]
    if base_cmd not in _SAFE_COMMANDS:
        return False

    if base_cmd in ("python", "python3"):
        return _is_safe_python_cmd(cmd_parts)
    if base_cmd == "pip":
        return _is_safe_pip_cmd(cmd_parts)

    allowed_args = _SAFE_COMMANDS[base_cmd]
    if not allowed_args:
        return True

    # 对于其他命令，仅校验“第一个非 flag 参数”（子命令）；其余参数放行
    subcmd = None
    for a in cmd_parts[1:]:
        if a.startswith("-"):
            continue
        subcmd = a
        break
    if subcmd is None:
        return True
    return any(allowed_arg == subcmd or allowed_arg.startswith(subcmd) for allowed_arg in allowed_args)

@utils.timer
@tool("git_status")
def git_status(porcelain: bool = True) -> str:
    """Get git status for WORKSPACE_ROOT (read-only)."""
    repo = utils.get_workspace_root()
    if not (repo / ".git").exists():
        return json.dumps({"ok": False, "error": "WORKSPACE_ROOT is not a git repo"}, ensure_ascii=False)
    args = ["git", "-C", str(repo), "status"]
    if porcelain:
        args += ["--porcelain=v1", "--untracked-files=all"]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        out = (p.stdout or "") + (p.stderr or "")
        return json.dumps({"ok": p.returncode == 0, "output": out}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "output": ""}, ensure_ascii=False)

@utils.timer
@tool("git_diff")
def git_diff(paths: List[str] = None, staged: bool = False, max_chars: int = 60000) -> str:
    """Get git diff (read-only). Optionally limit to paths under WORKSPACE_ROOT."""
    repo = utils.get_workspace_root()
    if not (repo / ".git").exists():
        return json.dumps({"ok": False, "error": "WORKSPACE_ROOT is not a git repo"}, ensure_ascii=False)
    args = ["git", "-C", str(repo), "diff"]
    if staged:
        args.append("--staged")
    args += ["--no-color"]
    safe_paths: List[str] = []
    if paths:
        for pth in paths:
            fp = _resolve_under_root(pth)
            safe_paths.append(str(fp.relative_to(repo)))
    if safe_paths:
        args.append("--")
        args.extend(safe_paths)
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        out = (p.stdout or "") + (p.stderr or "")
        if len(out) > int(max_chars):
            out = out[: int(max_chars)] + "\n...[truncated]"
        return json.dumps({"ok": p.returncode == 0, "output": out}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "output": ""}, ensure_ascii=False)

@utils.timer
@tool("git_show")
def git_show(spec: str = "HEAD", max_chars: int = 60000) -> str:
    """Show a commit or object (read-only). Example: HEAD, HEAD~1, <sha>."""
    repo = utils.get_workspace_root()
    if not (repo / ".git").exists():
        return json.dumps({"ok": False, "error": "WORKSPACE_ROOT is not a git repo"}, ensure_ascii=False)
    if not re.match(r"^[A-Za-z0-9_./~^:-]{1,80}$", spec or ""):
        return json.dumps({"ok": False, "error": "invalid git spec"}, ensure_ascii=False)
    args = ["git", "-C", str(repo), "show", "--no-color", "--stat", "--patch", spec]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        out = (p.stdout or "") + (p.stderr or "")
        if len(out) > int(max_chars):
            out = out[: int(max_chars)] + "\n...[truncated]"
        return json.dumps({"ok": p.returncode == 0, "output": out}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "output": ""}, ensure_ascii=False)

def _run_safe_command(command: str, timeout: int = 30) -> str:
    """
    在workspace内运行安全的命令（如python, pytest, black等）。
    只允许执行预定义的安全命令列表，防止任意命令执行。

    Args:
        command: 要执行的命令字符串，如 "python -m pytest tests/"
        timeout: 命令超时时间（秒），默认30秒

    Returns:
        JSON字符串包含执行结果：{"ok": bool, "stdout": str, "stderr": str, "exit_code": int}
    """
    import shlex

    try:
        cmd_parts = shlex.split(command.strip())
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "error": f"命令解析失败: {e}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            },
            ensure_ascii=False,
        )

    if not _is_safe_command(cmd_parts):
        return json.dumps(
            {
                "ok": False,
                "error": f"命令不在安全列表中: {cmd_parts[0]}",
                "allowed_commands": list(_SAFE_COMMANDS.keys()),
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            },
            ensure_ascii=False,
        )

    # 确保在workspace目录下执行
    cwd = utils.get_workspace_root()
    if not cwd.exists():
        return json.dumps(
            {
                "ok": False,
                "error": f"workspace目录不存在: {cwd}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            },
            ensure_ascii=False,
        )

    try:
        process = subprocess.run(
            cmd_parts,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return json.dumps(
            {
                "ok": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "exit_code": process.returncode,
            },
            ensure_ascii=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "ok": False,
                "error": f"命令执行超时 ({timeout}秒)",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "error": f"命令执行失败: {type(e).__name__}: {e}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
            },
            ensure_ascii=False,
        )

@utils.timer
@tool("run_safe_command")
def run_safe_command(command: str, timeout: int = 30) -> str:
    """在workspace内运行安全的命令（如python, pytest, black等）。只允许执行预定义的安全命令列表，防止任意命令执行。"""
    return _run_safe_command(command, timeout)

@utils.timer
@tool("analyze_code")
def analyze_code(path: str = ".", tool_name: str = "flake8") -> str:
    """
    运行代码静态分析工具（flake8或pylint）。

    Args:
        path: 要分析的文件或目录路径（相对于workspace）
        tool_name: 分析工具，可选 "flake8" 或 "pylint"

    Returns:
        JSON字符串包含分析结果
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps(
            {"ok": False, "error": f"路径不存在: {path}", "output": ""},
            ensure_ascii=False,
        )

    if tool_name == "flake8":
        cmd = ["flake8", str(fp)]
    elif tool_name == "pylint":
        cmd = ["pylint", str(fp)]
    else:
        return json.dumps(
            {
                "ok": False,
                "error": f"不支持的工具: {tool_name}，可选 flake8 或 pylint",
                "output": "",
            },
            ensure_ascii=False,
        )

    result = _run_safe_command(" ".join(cmd), timeout=60)
    result_dict = json.loads(result)

    # 如果run_safe_command失败，返回错误
    if not result_dict.get("ok", False):
        return json.dumps(
            {
                "ok": False,
                "error": result_dict.get("error", "分析失败"),
                "output": result_dict.get("stderr", "") + result_dict.get("stdout", ""),
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "ok": True,
            "output": result_dict.get("stdout", ""),
            "tool": tool_name,
            "path": str(fp.relative_to(utils.get_workspace_root())),
        },
        ensure_ascii=False,
    )

@utils.timer
@tool("run_tests")
def run_tests(path: str = "tests", options: str = "-v") -> str:
    """
    运行pytest测试。

    Args:
        path: 测试目录或文件路径（相对于workspace）
        options: pytest选项，如 "-v" 详细输出，"--tb=short" 简短回溯

    Returns:
        JSON字符串包含测试结果
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps(
            {"ok": False, "error": f"测试路径不存在: {path}", "output": ""},
            ensure_ascii=False,
        )

    cmd = f"pytest {str(fp)} {options}".strip()
    result = _run_safe_command(cmd, timeout=120)
    result_dict = json.loads(result)

    # 提取测试统计信息
    output = result_dict.get("stdout", "")
    stats = {}
    for line in output.splitlines():
        if "passed" in line and "failed" in line and "error" in line:
            # 尝试解析 pytest 统计行
            import re

            match = re.search(r"(\d+) passed.*?(\d+) failed.*?(\d+) error", line)
            if match:
                stats = {
                    "passed": int(match.group(1)),
                    "failed": int(match.group(2)),
                    "error": int(match.group(3)),
                }
            break

    return json.dumps(
        {
            "ok": result_dict.get("ok", False),
            "exit_code": result_dict.get("exit_code", -1),
            "output": output,
            "stats": stats,
            "path": str(fp.relative_to(utils.get_workspace_root())),
        },
        ensure_ascii=False,
    )

@utils.timer
@tool("format_code")
def format_code(path: str = ".", formatter: str = "black") -> str:
    """
    使用代码格式化工具（black或isort）格式化代码。

    Args:
        path: 要格式化的文件或目录路径（相对于workspace）
        formatter: 格式化工具，可选 "black" 或 "isort"

    Returns:
        JSON字符串包含格式化结果
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps(
            {"ok": False, "error": f"路径不存在: {path}", "output": ""},
            ensure_ascii=False,
        )

    if formatter == "black":
        cmd = f"black {str(fp)}"
    elif formatter == "isort":
        cmd = f"isort {str(fp)}"
    else:
        return json.dumps(
            {
                "ok": False,
                "error": f"不支持的工具: {formatter}，可选 black 或 isort",
                "output": "",
            },
            ensure_ascii=False,
        )

    result = _run_safe_command(cmd, timeout=60)
    result_dict = json.loads(result)

    return json.dumps(
        {
            "ok": result_dict.get("ok", False),
            "output": result_dict.get("stdout", "") + result_dict.get("stderr", ""),
            "formatter": formatter,
            "path": str(fp.relative_to(utils.get_workspace_root())),
        },
        ensure_ascii=False,
    )

@utils.timer
@tool("check_syntax")
def check_syntax(path: str) -> str:
    """
    检查Python文件的语法是否正确。

    Args:
        path: Python文件路径（相对于workspace）

    Returns:
        JSON字符串包含语法检查结果
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps(
            {"ok": False, "error": f"文件不存在: {path}", "output": ""},
            ensure_ascii=False,
        )

    if fp.suffix != ".py":
        return json.dumps(
            {"ok": False, "error": f"不是Python文件: {path}", "output": ""},
            ensure_ascii=False,
        )

    cmd = f"python -m py_compile {str(fp)}"
    result = _run_safe_command(cmd, timeout=30)
    result_dict = json.loads(result)

    return json.dumps(
        {
            "ok": result_dict.get("ok", False),
            "output": result_dict.get("stdout", "") + result_dict.get("stderr", ""),
            "path": str(fp.relative_to(utils.get_workspace_root())),
        },
        ensure_ascii=False,
    )

@utils.timer
@tool("extract_functions")
def extract_functions(path: str) -> str:
    """
    从Python文件中提取函数签名和文档字符串。

    Args:
        path: Python文件路径（相对于workspace）

    Returns:
        JSON字符串包含提取的函数信息列表
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps(
            {"ok": False, "error": f"文件不存在: {path}", "functions": []},
            ensure_ascii=False,
        )

    try:
        content = fp.read_text(encoding="utf-8")
        tree = ast.parse(content)

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node) or "",
                    "decorators": (
                        [ast.unparse(dec) for dec in node.decorator_list]
                        if hasattr(ast, "unparse")
                        else []
                    ),
                }
                functions.append(func_info)

        return json.dumps(
            {
                "ok": True,
                "functions": functions,
                "count": len(functions),
                "path": str(fp.relative_to(utils.get_workspace_root())),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "error": f"解析失败: {type(e).__name__}: {e}",
                "functions": [],
            },
            ensure_ascii=False,
        )

@utils.timer
@tool("get_code_metrics")
def get_code_metrics(path: str = ".") -> str:
    """
    获取代码度量信息（文件数、行数、平均复杂度等）。

    Args:
        path: 目录或文件路径（相对于workspace）

    Returns:
        JSON字符串包含代码度量信息
    """
    import ast
    from pathlib import Path

    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps(
            {"ok": False, "error": f"路径不存在: {path}", "metrics": {}},
            ensure_ascii=False,
        )

    files = []
    if fp.is_file():
        files = [fp]
    else:
        files = list(fp.rglob("*.py"))

    total_lines = 0
    total_functions = 0
    total_classes = 0
    file_metrics = []

    for py_file in files:
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.count("\n") + 1
            total_lines += lines

            tree = ast.parse(content)
            functions = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]
            classes = [
                node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]

            total_functions += len(functions)
            total_classes += len(classes)

            file_metrics.append(
                {
                    "file": str(py_file.relative_to(utils.get_workspace_root())),
                    "lines": lines,
                    "functions": len(functions),
                    "classes": len(classes),
                }
            )
        except Exception:
            continue

    avg_functions_per_file = total_functions / len(files) if files else 0
    avg_lines_per_file = total_lines / len(files) if files else 0

    return json.dumps(
        {
            "ok": True,
            "metrics": {
                "total_files": len(files),
                "total_lines": total_lines,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "avg_functions_per_file": round(avg_functions_per_file, 2),
                "avg_lines_per_file": round(avg_lines_per_file, 2),
            },
            "file_details": file_metrics,
            "path": str(fp.relative_to(utils.get_workspace_root())),
        },
        ensure_ascii=False,
    )

@utils.timer
@tool("save_mermaid_diagram")
def save_mermaid_diagram(mermaid_code: str, filename: str = "diagram.md") -> str:
    """
    Save Mermaid diagram code to a file in the workspace.

    Args:
        mermaid_code: The Mermaid diagram code (e.g., flowchart, sequence diagram)
        filename: Output filename (should end with .md for Markdown rendering)

    Returns:
        JSON string with success status and file path
    """
    import json

    fp = _resolve_under_root(filename)
    try:
        # Ensure the directory exists
        fp.parent.mkdir(parents=True, exist_ok=True)

        # Create a Markdown file with Mermaid code block
        content = f"```mermaid\n{mermaid_code}\n```"
        fp.write_text(content, encoding="utf-8")

        return json.dumps(
            {
                "ok": True,
                "path": str(fp.relative_to(utils.get_workspace_root())),
                "message": f"Mermaid diagram saved to {fp.relative_to(utils.get_workspace_root())}",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "ok": False,
                "error": type(e).__name__,
                "reason": str(e),
                "path": filename,
            },
            ensure_ascii=False,
        )

@utils.timer
@tool("generate_diagram_description")
def generate_diagram_description(
        description: str, diagram_type: str = "flowchart"
) -> str:
    """
    Generate a detailed diagram description that can be used to create Mermaid code.
    This tool helps structure the natural language description for better diagram generation.

    Args:
        description: Natural language description of the diagram
        diagram_type: Type of diagram (flowchart, sequenceDiagram, classDiagram, stateDiagram, gantt, pie)

    Returns:
        JSON string with structured description and suggestions
    """
    import json

    # This is a placeholder that returns the description structured
    # In a real implementation, you might use an LLM to refine the description
    return json.dumps(
        {
            "ok": True,
            "diagram_type": diagram_type,
            "structured_description": description,
            "suggestions": [
                "Use the save_mermaid_diagram tool to save generated Mermaid code",
                "Common Mermaid syntax: flowchart TD for top-down flowcharts",
                "For sequence diagrams: participant A, participant B, A->B: message",
            ],
        },
        ensure_ascii=False,
    )

@utils.timer
@tool("create_plotly_chart")
def create_plotly_chart(
        chart_type: str,
        data: str,
        title: str = "",
        filename: str = "chart.html",
        x_label: str = "",
        y_label: str = "",
        width: int = 800,
        height: int = 600,
) -> str:
    """
    Create a Plotly chart and save as HTML file.

    Args:
        chart_type: Type of chart - "line", "bar", "scatter", "pie", "histogram"
        data: JSON string containing chart data. Format depends on chart type:
            - For line/bar/scatter: {"x": [values], "y": [values]} or
              {"series": [{"x": [...], "y": [...], "name": "series1"}, ...]}
            - For pie: {"labels": [...], "values": [...]}
            - For histogram: {"values": [...]}
        title: Chart title
        filename: Output HTML filename
        x_label: X-axis label
        y_label: Y-axis label
        width: Chart width in pixels
        height: Chart height in pixels

    Returns:
        JSON string with success status and file path
    """
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fp = _resolve_under_root(filename)
    try:
        # Parse input data
        data_dict = json.loads(data)

        # Create figure based on chart type
        fig = None

        if chart_type == "line":
            if "series" in data_dict:
                fig = go.Figure()
                for series in data_dict["series"]:
                    fig.add_trace(
                        go.Scatter(
                            x=series.get("x", []),
                            y=series.get("y", []),
                            mode="lines+markers",
                            name=series.get("name", ""),
                        )
                    )
            else:
                fig = go.Figure(
                    data=go.Scatter(
                        x=data_dict.get("x", []),
                        y=data_dict.get("y", []),
                        mode="lines+markers",
                    )
                )

        elif chart_type == "bar":
            if "series" in data_dict:
                fig = go.Figure()
                for series in data_dict["series"]:
                    fig.add_trace(
                        go.Bar(
                            x=series.get("x", []),
                            y=series.get("y", []),
                            name=series.get("name", ""),
                        )
                    )
            else:
                fig = go.Figure(
                    data=go.Bar(x=data_dict.get("x", []), y=data_dict.get("y", []))
                )

        elif chart_type == "scatter":
            if "series" in data_dict:
                fig = go.Figure()
                for series in data_dict["series"]:
                    fig.add_trace(
                        go.Scatter(
                            x=series.get("x", []),
                            y=series.get("y", []),
                            mode="markers",
                            name=series.get("name", ""),
                        )
                    )
            else:
                fig = go.Figure(
                    data=go.Scatter(
                        x=data_dict.get("x", []),
                        y=data_dict.get("y", []),
                        mode="markers",
                    )
                )

        elif chart_type == "pie":
            fig = go.Figure(
                data=go.Pie(
                    labels=data_dict.get("labels", []),
                    values=data_dict.get("values", []),
                )
            )

        elif chart_type == "histogram":
            fig = go.Figure(data=go.Histogram(x=data_dict.get("values", [])))

        else:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Unsupported chart type",
                    "supported_types": ["line", "bar", "scatter", "pie", "histogram"],
                },
                ensure_ascii=False,
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width,
            height=height,
            showlegend=True if chart_type in ["line", "bar", "scatter"] else False,
        )

        # Ensure directory exists
        fp.parent.mkdir(parents=True, exist_ok=True)

        # Save as HTML
        fig.write_html(str(fp))

        return json.dumps(
            {
                "ok": True,
                "path": str(fp.relative_to(utils.get_workspace_root())),
                "filename": filename,
                "chart_type": chart_type,
                "message": f"Plotly {chart_type} chart saved to {fp.relative_to(utils.get_workspace_root())}",
            },
            ensure_ascii=False,
        )

    except json.JSONDecodeError as e:
        return json.dumps(
            {"ok": False, "error": "Invalid JSON data", "reason": str(e)},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"ok": False, "error": type(e).__name__, "reason": str(e)},
            ensure_ascii=False,
        )

@utils.timer
@tool("save_chart_data")
def save_chart_data(
        data: str, filename: str = "chart_data.json", format: str = "json"
) -> str:
    """
    Save chart data to a file in CSV or JSON format.

    Args:
        data: JSON string containing chart data
        filename: Output filename (with .json or .csv extension)
        format: Output format - "json" or "csv"

    Returns:
        JSON string with success status and file path
    """
    import json
    import pandas as pd

    fp = _resolve_under_root(filename)
    try:
        # Parse input data
        data_dict = json.loads(data)

        # Ensure directory exists
        fp.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "csv":
            # Convert to DataFrame and save as CSV
            # Handle different data structures
            if isinstance(data_dict, dict):
                # Try to create DataFrame from dict
                df = pd.DataFrame(data_dict)
            elif isinstance(data_dict, list):
                df = pd.DataFrame(data_dict)
            else:
                # Wrap in a single column
                df = pd.DataFrame({"data": [data_dict]})

            df.to_csv(fp, index=False, encoding="utf-8")

        else:  # JSON format
            # Save as pretty JSON
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)

        return json.dumps(
            {
                "ok": True,
                "path": str(fp.relative_to(utils.get_workspace_root())),
                "filename": filename,
                "format": format,
                "message": f"Chart data saved to {fp.relative_to(utils.get_workspace_root())}",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps(
            {"ok": False, "error": type(e).__name__, "reason": str(e)},
            ensure_ascii=False,
        )

@utils.timer
@tool("analyze_data_for_chart")
def analyze_data_for_chart(data: str, description: str = "") -> str:
    """
    Analyze data and suggest appropriate chart types and configurations.

    Args:
        data: JSON string containing data to analyze
        description: Optional description of what the data represents

    Returns:
        JSON string with analysis and recommendations
    """
    import json
    import pandas as pd
    import numpy as np

    try:
        # Parse input data
        data_dict = json.loads(data)

        # Convert to DataFrame for analysis
        df = None
        if isinstance(data_dict, dict):
            # Check if it's a simple key-value structure
            if all(isinstance(v, list) for v in data_dict.values()):
                # Multiple lists
                df = pd.DataFrame(data_dict)
            else:
                # Single series
                df = pd.DataFrame(
                    {"value": list(data_dict.values())}, index=list(data_dict.keys())
                )
        elif isinstance(data_dict, list):
            df = pd.DataFrame(data_dict)

        analysis = {
            "data_shape": None,
            "data_types": None,
            "summary_stats": None,
            "recommended_charts": [],
            "configuration_suggestions": {},
        }

        if df is not None:
            # Basic analysis
            analysis["data_shape"] = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns),
            }

            analysis["data_types"] = {col: str(df[col].dtype) for col in df.columns}

            # Summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["summary_stats"] = df[numeric_cols].describe().to_dict()

            # Chart recommendations based on data characteristics
            recommendations = []

            if len(df.columns) >= 2:
                # At least two columns - can do line, bar, scatter
                recommendations.append(
                    {
                        "type": "line",
                        "reason": "Suitable for showing trends over time or ordered categories",
                        "suggested_config": {
                            "x": df.columns[0],
                            "y": (
                                df.columns[1] if len(df.columns) > 1 else df.columns[0]
                            ),
                        },
                    }
                )

                recommendations.append(
                    {
                        "type": "bar",
                        "reason": "Good for comparing discrete categories",
                        "suggested_config": {
                            "x": df.columns[0],
                            "y": (
                                df.columns[1] if len(df.columns) > 1 else df.columns[0]
                            ),
                        },
                    }
                )

                if len(df) > 10:  # Enough points for scatter
                    recommendations.append(
                        {
                            "type": "scatter",
                            "reason": "Useful for showing relationships between two variables",
                            "suggested_config": {
                                "x": df.columns[0],
                                "y": (
                                    df.columns[1]
                                    if len(df.columns) > 1
                                    else df.columns[0]
                                ),
                            },
                        }
                    )

            # Check for categorical data
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                recommendations.append(
                    {
                        "type": "pie",
                        "reason": "Good for showing proportions of categorical data",
                        "suggested_config": {
                            "labels": (
                                df[categorical_cols[0]].tolist()
                                if len(categorical_cols) > 0
                                else []
                            ),
                            "values": (
                                df[numeric_cols[0]].tolist()
                                if len(numeric_cols) > 0
                                else []
                            ),
                        },
                    }
                )

            # Check for single numeric column (histogram)
            if len(numeric_cols) == 1:
                recommendations.append(
                    {
                        "type": "histogram",
                        "reason": "Shows distribution of a single numeric variable",
                        "suggested_config": {"values": df[numeric_cols[0]].tolist()},
                    }
                )

            analysis["recommended_charts"] = recommendations

            # Configuration suggestions
            if title := description:
                analysis["configuration_suggestions"]["title"] = title

            if len(df.columns) >= 2:
                analysis["configuration_suggestions"]["x_label"] = str(df.columns[0])
                analysis["configuration_suggestions"]["y_label"] = (
                    str(df.columns[1]) if len(df.columns) > 1 else "Value"
                )

        return json.dumps(
            {
                "ok": True,
                "analysis": analysis,
                "description": description,
                "suggestions": [
                    "Use create_plotly_chart to generate the chart",
                    "Use save_chart_data to save the raw data",
                    "Consider adding a title and axis labels for clarity",
                ],
            },
            ensure_ascii=False,
        )

    except json.JSONDecodeError as e:
        return json.dumps(
            {"ok": False, "error": "Invalid JSON data", "reason": str(e)},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"ok": False, "error": type(e).__name__, "reason": str(e)},
            ensure_ascii=False,
        )


# ============================================================================
# 通用源码分析工具 (支持 C++, Go, Python, Java, etc.)
# ============================================================================

COMMON_SOURCE_EXTENSIONS = [
    # C/C++
    ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx", ".hxx", ".m", ".mm",
    # Python
    ".py", ".pyi",
    # Go
    ".go",
    # Java/Kotlin/Scala
    ".java", ".kt", ".kts", ".scala",
    # Web
    ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".vue", ".svelte",
    # Rust
    ".rs",
    # Shell
    ".sh", ".bash", ".zsh",
    # Others
    ".rb", ".php", ".cs", ".swift", ".lua", ".pl", ".pm"
]

@utils.timer
@tool("read_source_file")
def read_source_file(path: str, start: int = 0, max_chars: int = 100 * 1024 * 1024) -> str:
    """读取源代码文件，支持多种语言 (C++, Go, Python, Java, TS 等)。支持大文件分块读取。"""
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    suffix = fp.suffix.lower()
    # 宽松检查：只要是文本文件通常都可以读取，但为了区分，这里检查常见源码后缀
    # 或者如果不在列表中但它是文件，我们也尝试读取，只是给个警告？
    # 为了保持行为，先检查后缀
    if suffix not in COMMON_SOURCE_EXTENSIONS and suffix not in [".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".xml", ".sql"]:
         # 如果不是常见源码，尝试读取前几个字节判断是否为二进制
         pass 

    try:
        data = fp.read_text(encoding="utf-8", errors="replace")
        if start < 0:
            start = 0
        content = data[start: start + max_chars]
        lines = data[:start + max_chars].count("\n") + 1
        return json.dumps({
            "ok": True,
            "path": str(fp.relative_to(utils.get_workspace_root())),
            "content": content,
            "total_chars": len(data),
            "total_lines": lines,
            "read_range": f"{start}-{start + max_chars}",
            "language": suffix[1:] if suffix.startswith(".") else "unknown"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


from typing import Dict, List, Any, Optional, Union


def _split_extensions_arg(extensions: Union[List[str], str]) -> List[str]:
    """将 extensions 入参容错拆分成 token 列表。

    支持：
    - [".py", ".go"]
    - ".py,.go" / "*.py,*.go" / "py go" / "py;go"
    """
    import re

    if isinstance(extensions, list):
        return [str(x).strip() for x in extensions if str(x).strip()]
    s = str(extensions).strip()
    if not s:
        return []
    return [t for t in re.split(r"[\s,;]+", s) if t]


def _normalize_extension_token(token: str) -> Optional[str]:
    """把 '*.py' / 'py' / '.py' 归一化为 '.py'。"""
    t = (token or "").strip().lower()
    if not t:
        return None

    # 去掉路径前缀（防止传入像 'src/*.py' 这种）
    t = t.split("/")[-1].split("\\")[-1]

    # 常见“全匹配”写法：当作未指定扩展
    if t in {"*", ".*", "all", "any"}:
        return None

    # '*.py' => '.py'
    if t.startswith("*.") and len(t) > 2:
        t = t[1:]

    # 若仍包含通配符，尽量提取最后一个 '.' 作为后缀
    if any(ch in t for ch in ("*", "?", "[", "]")):
        if "." in t:
            t = t[t.rfind(".") :]
        else:
            return None

    if not t.startswith("."):
        t = "." + t
    return None if t == "." else t


def _normalize_extensions(
    extensions: Optional[Union[List[str], str]],
    default_exts: List[str],
) -> List[str]:
    """extensions 归一化；若结果为空，则回退到 default_exts。"""
    if not extensions:
        return default_exts

    tokens = _split_extensions_arg(extensions)
    norm: List[str] = []
    seen = set()
    for tok in tokens:
        e = _normalize_extension_token(tok)
        if not e:
            continue
        if e not in seen:
            seen.add(e)
            norm.append(e)
    return norm if norm else default_exts

@utils.timer
@tool("list_source_files")
def list_source_files(path: str, max_files: int = 500, extensions: Optional[Union[List[str], str]] = None) -> str:
    """递归列出目录下的源代码文件。
    
    Args:
        path: 目录路径
        max_files: 最大返回数量
        extensions: 可选，指定后缀列表 (e.g. [".go", ".py"] or ".go,.py")。如果不传，默认匹配所有常见源码后缀。
    """
    import logging
    logger = logging.getLogger(__name__)
    
    root = _get_search_root(path)
    
    # 确定要搜索的后缀（支持 '*.py' 这类通配写法）
    target_exts = _normalize_extensions(extensions, COMMON_SOURCE_EXTENSIONS)

    source_files = []

    # 遍历
    # 为提高效率，如果 extensions 较少可以使用 glob，否则遍历所有文件检查后缀
    for fp in root.rglob("*"):
        if len(source_files) >= max_files:
            break
        if not fp.is_file():
            continue
        
        if fp.suffix.lower() in target_exts:
             try:
                rel_path = str(fp.relative_to(utils.get_workspace_root()))
                source_files.append(rel_path)
             except ValueError:
                continue

    result = {
        "ok": True,
        "path": str(root),
        "count": len(source_files),
        "extensions": target_exts if extensions else "all_common"
    }

    if len(source_files) > 100:
        # 保存到文件
        import time
        ts = int(time.time())
        list_filename = f"source_files_list_{ts}.json"
        list_path = utils.get_workspace_root() / "tmp" / list_filename
        (utils.get_workspace_root() / "tmp").mkdir(parents=True, exist_ok=True)

        with open(list_path, "w", encoding="utf-8") as f:
            json.dump(source_files, f, ensure_ascii=False, indent=2)

        try:
            result["files_list_path"] = str(list_path.relative_to(utils.get_workspace_root()))
        except ValueError:
            result["files_list_path"] = str(list_path)

        result["message"] = f"文件数量较多，完整列表已保存至 {result['files_list_path']}。"
        result["files_preview"] = source_files[:20]
    else:
        result["files"] = source_files

    return json.dumps(result, ensure_ascii=False)

@utils.timer
@tool("grep_source_code")
def grep_source_code(pattern: str, path: str, max_matches: int = 100, extensions: Optional[Union[List[str], str]] = None) -> str:
    """在源代码文件中搜索正则表达式。
    
    Args:
        pattern: 正则
        path: 路径
        max_matches: 最大匹配数
        extensions: 可选后缀列表
    """
    import re
    root = _get_search_root(path)
    
    target_exts = _normalize_extensions(extensions, COMMON_SOURCE_EXTENSIONS)

    try:
        rx = re.compile(pattern)
    except re.error as e:
        return json.dumps({"ok": False, "error": f"无效正则: {e}"}, ensure_ascii=False)

    matches = []
    for fp in root.rglob("*"):
        if len(matches) >= max_matches:
            break
        if not fp.is_file() or fp.suffix.lower() not in target_exts:
            continue
        try:
            # 限制文件大小，避免 grep 极大的文件
            if fp.stat().st_size > 10 * 1024 * 1024:
                continue
                
            for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if rx.search(line):
                    try:
                        rel_path = str(fp.relative_to(utils.get_workspace_root()))
                    except ValueError:
                        rel_path = str(fp)
                    matches.append({
                        "file": rel_path,
                        "line": i,
                        "content": line.strip()[:200]
                    })
        except Exception:
            continue

    return json.dumps({
        "ok": True,
        "pattern": pattern,
        "count": len(matches),
        "matches": matches[:max_matches]
    }, ensure_ascii=False)

@utils.timer
@tool("extract_cpp_functions")
def extract_cpp_functions(path: str) -> str:
    """从 C/C++ 文件中提取函数签名、类和方法。"""
    import re

    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    content = fp.read_text(encoding="utf-8", errors="replace")
    functions = []
    classes = []

    # 提取函数定义 (返回类型 函数名(参数))
    func_pattern = re.compile(
        r'^\s*(inline\s+)?(static\s+)?(constexpr\s+)?([\w:\*&\s]+?)\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(const)?\s*(override)?\s*(noexcept)?\s*\{',
        re.MULTILINE
    )

    # 提取类/结构体定义
    class_pattern = re.compile(
        r'^\s*(template\s*<[^>]*>\s+)?(class|struct)\s+([a-zA-Z_]\w*)',
        re.MULTILINE
    )

    # 提取 #include 语句
    include_pattern = re.compile(r'#include\s*[<"]([^>"]+)[>"]')

    for i, line in enumerate(content.splitlines(), 1):
        func_match = func_pattern.search(line)
        if func_match:
            functions.append({
                "line": i,
                "name": func_match.group(5),
                "return_type": func_match.group(4).strip(),
                "params": func_match.group(6).strip(),
                "const": bool(func_match.group(7)),
                "override": bool(func_match.group(8))
            })

        class_match = class_pattern.search(line)
        if class_match:
            classes.append({
                "line": i,
                "name": class_match.group(3),
                "template": bool(class_match.group(1))
            })

    includes = include_pattern.findall(content)

    return json.dumps({
        "ok": True,
        "path": str(fp.relative_to(utils.get_workspace_root())),
        "functions": functions,
        "classes": classes,
        "includes": list(set(includes))[:20]
    }, ensure_ascii=False)

@utils.timer
@tool("get_source_metrics")
def get_source_metrics(path: str) -> str:
    """获取项目的代码度量信息 (Lines of Code, Files, etc.)。"""
    # 如果 path 为空或 "."，使用 WORKSPACE.ROOT
    if not path or path == ".":
        path = utils.env("WORKSPACE.ROOT", default=".")

    root = _resolve_under_root(path)
    
    source_files = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in COMMON_SOURCE_EXTENSIONS:
            source_files.append(fp)

    total_lines = 0
    total_code_lines = 0
    # 简单的计数
    total_functions_approx = 0 
    total_classes_approx = 0
    files_info = []

    for fp in source_files:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines += len(lines)

            # 简单估算代码行（排除空行和注释行）
            # 注意：不同语言注释不同，这里仅处理常见的 // 和 #
            code_lines_count = 0
            for l in lines:
                s = l.strip()
                if not s:
                    continue
                if s.startswith("//") or s.startswith("#"):
                    continue
                code_lines_count += 1
            
            total_code_lines += code_lines_count

            # 简单启发式统计
            # C/C++/Java/Go/Rust: braces
            # Python: def/class
            suffix = fp.suffix.lower()
            if suffix in [".py"]:
                total_functions_approx += content.count("def ")
                total_classes_approx += content.count("class ")
            else:
                # C-style
                total_functions_approx += max(0, content.count("{") // 3) # 非常粗略
                total_classes_approx += content.count("class ") + content.count("struct ") + content.count("interface ")

            files_info.append({
                "file": str(fp.relative_to(utils.get_workspace_root())),
                "lines": len(lines),
                "code_lines": code_lines_count
            })
        except Exception:
            continue

    return json.dumps({
        "ok": True,
        "total_files": len(source_files),
        "total_lines": total_lines,
        "code_lines": total_code_lines,
        "estimated_functions": total_functions_approx,
        "estimated_classes": total_classes_approx,
        "files": files_info[:100]
    }, ensure_ascii=False)

@utils.timer
@tool("run_cpp_linter")
def run_cpp_linter(path: str, tool_name: str = "clang-tidy") -> str:
    """运行 C++ 静态分析工具 (clang-tidy, cppcheck)。

    Args:
        path: C++ 文件或目录路径
        tool_name: 分析工具，可选 clang-tidy, cppcheck
    """
    # 如果 path 为空或 "."，使用 WORKSPACE.ROOT
    if not path or path == ".":
        path = utils.env("WORKSPACE.ROOT", default=".")

    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"路径不存在: {path}"}, ensure_ascii=False)

    # 检查可用工具
    available_tools = []
    for cmd in ["clang-tidy", "cppcheck", "clang-format"]:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        if result.returncode == 0:
            available_tools.append(cmd)

    if tool_name not in available_tools:
        return json.dumps({
            "ok": False,
            "error": f"工具 {tool_name} 不可用",
            "available_tools": available_tools
        }, ensure_ascii=False)

    try:
        if tool_name == "clang-tidy":
            cmd = ["clang-tidy", str(fp), "--", "-std=c++17", "-I", str(fp.parent)]
        elif tool_name == "cppcheck":
            cmd = ["cppcheck", "--enable=all", str(fp)]
        else:
            cmd = ["clang-format", str(fp)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return json.dumps({
            "ok": True,
            "tool": tool_name,
            "path": str(fp.relative_to(utils.get_workspace_root())),
            "output": result.stdout[:10000] + ("..." if len(result.stdout) > 10000 else ""),
            "errors": result.stderr[:5000] if result.stderr else ""
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "分析超时"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("check_cpp_syntax")
def check_cpp_syntax(path: str, compiler: str = "clang++", std: str = "c++17") -> str:
    """通过编译检查 C/C++ 语法正确性。"""
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    suffix = fp.suffix.lower()
    if suffix not in [".c", ".cpp", ".cc", ".cxx"]:
        return json.dumps({"ok": False, "error": f"不是 C++ 源文件: {path}"}, ensure_ascii=False)

    cmd = [compiler, "-fsyntax-only", f"-std={std}", str(fp)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.dumps({
            "ok": result.returncode == 0,
            "compiler": compiler,
            "path": str(fp.relative_to(utils.get_workspace_root())),
            "output": result.stderr if result.stderr else "编译检查通过",
            "exit_code": result.returncode
        }, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"ok": False, "error": f"编译器 {compiler} 不存在"}, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "编译超时"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("format_cpp")
def format_cpp(path: str, style: str = "llvm") -> str:
    """使用 clang-format 格式化 C/C++ 代码。"""
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    suffix = fp.suffix.lower()
    if suffix not in [".c", ".cpp", ".h", ".hpp", ".cc", ".cxx", ".hxx"]:
        return json.dumps({"ok": False, "error": f"不是 C/C++ 文件: {path}"}, ensure_ascii=False)

    cmd = ["clang-format", f"-style={style}", str(fp)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            # 备份并写入格式化后的内容
            backup = fp.with_suffix(fp.suffix + ".bak")
            shutil.copy2(fp, backup)

            with open(fp, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            return json.dumps({
                "ok": True,
                "path": str(fp.relative_to(utils.get_workspace_root())),
                "style": style,
                "backup": str(backup.relative_to(utils.get_workspace_root())),
                "message": "代码已格式化"
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "ok": False,
                "error": result.stderr
            }, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"ok": False, "error": "clang-format 未安装"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("find_cmake_files")
def find_cmake_files(path: str) -> str:
    """查找 CMake 项目文件 (CMakeLists.txt, *.cmake)。

    Args:
        path: 项目目录路径（相对于 WORKSPACE_ROOT）
    """
    # 如果 path 为空或 "."，使用 WORKSPACE.ROOT
    if not path or path == ".":
        path = utils.env("WORKSPACE.ROOT", default=".")

    root = _resolve_under_root(path)
    cmake_files = []

    for fp in sorted(root.rglob("*")):
        if fp.is_file():
            name = fp.name
            if name == "CMakeLists.txt" or name.endswith(".cmake"):
                cmake_files.append(str(fp.relative_to(utils.get_workspace_root())))

    # 查找 compile_commands.json
    compile_db = root / "compile_commands.json"
    has_compile_db = compile_db.exists()

    return json.dumps({
        "ok": True,
        "path": str(root.relative_to(utils.get_workspace_root())),
        "cmake_files": cmake_files,
        "has_compile_commands": has_compile_db,
        "count": len(cmake_files)
    }, ensure_ascii=False)

@utils.timer
@tool("read_compile_commands")
def read_compile_commands(path: str = "compile_commands.json", json=None) -> str:
    """读取编译数据库文件，分析项目构建配置。"""
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    try:
        import json
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取编译命令摘要
        commands = []
        for entry in data[:50]:
            commands.append({
                "file": entry.get("file", ""),
                "directory": entry.get("directory", ""),
                "command": entry.get("command", "")[:500]
            })

        # 统计编译器类型
        compilers = set()
        for entry in data:
            cmd = entry.get("command", "")
            if "clang++" in cmd:
                compilers.add("clang++")
            elif "g++" in cmd:
                compilers.add("g++")
            elif "c++" in cmd:
                compilers.add("c++")

        return json.dumps({
            "ok": True,
            "path": str(fp.relative_to(utils.get_workspace_root())),
            "entries_count": len(data),
            "compilers": list(compilers),
            "sample_commands": commands
        }, ensure_ascii=False)
    except json.JSONDecodeError:
        return json.dumps({"ok": False, "error": "无效的 JSON 文件"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# @tool("batch_read_source_files")
# def batch_read_source_files(paths: List[str], max_chars_per_file: int = 50000) -> str:
#     """批量读取多个源代码文件。
#
#     Args:
#         paths: 文件路径列表（相对于 WORKSPACE_ROOT）
#         max_chars_per_file: 每个文件最大读取字符数，默认 50000
#
#     Returns:
#         JSON 字符串包含所有文件的内容
#     """
#     results = []
#
#     for path in paths:
#         fp = _resolve_under_root(path)
#         if not fp.exists():
#             results.append({
#                 "path": path,
#                 "ok": False,
#                 "error": "文件不存在"
#             })
#             continue
#
#         suffix = fp.suffix.lower()
#         if suffix not in COMMON_SOURCE_EXTENSIONS:
#             # 尝试宽松读取
#              pass
#
#         try:
#             content = fp.read_text(encoding="utf-8", errors="replace")
#             results.append({
#                 "path": path,
#                 "ok": True,
#                 "content": content[:max_chars_per_file],
#                 "total_chars": len(content),
#                 "truncated": len(content) > max_chars_per_file
#             })
#         except Exception as e:
#             results.append({
#                 "path": path,
#                 "ok": False,
#                 "error": str(e)
#             })
#
#     return json.dumps({
#         "ok": True,
#         "total_files": len(paths),
#         "success_count": sum(1 for r in results if r.get("ok")),
#         "results": results
#     }, ensure_ascii=False)


# ============================================================================
# Bash 脚本执行工具
# ============================================================================

@utils.timer
@tool("run_bash")
def run_bash(command: str, timeout: int = 60, work_dir: str = ".") -> str:
    """执行 bash 命令。

    Args:
        command: 要执行的 bash 命令
        timeout: 超时时间（秒），默认 60
        work_dir: 工作目录（相对于 WORKSPACE_ROOT），默认当前目录

    Returns:
        JSON 字符串包含执行结果
    """
    # 如果 work_dir 为空或 "."，使用 WORKSPACE.ROOT
    if not work_dir or work_dir == ".":
        work_dir = utils.env("WORKSPACE.ROOT", default=".")

    work_path = _resolve_under_root(work_dir)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(work_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
        return json.dumps({
            "ok": result.returncode == 0,
            "command": command,
            "work_dir": str(work_path.relative_to(utils.get_workspace_root())),
            "exit_code": result.returncode,
            "stdout": result.stdout[:50000] if result.stdout else "",
            "stderr": result.stderr[:10000] if result.stderr else ""
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "ok": False,
            "command": command,
            "error": f"命令执行超时 ({timeout}秒)",
            "timeout": timeout
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "ok": False,
            "command": command,
            "error": str(e)
        }, ensure_ascii=False)


# ============================================================================
# 目录访问工具
# ============================================================================
@utils.timer
@tool("get_file_info")
def get_file_info(path: str) -> str:
    """获取文件或目录的详细信息。

    Args:
        path: 文件或目录路径（相对于 WORKSPACE_ROOT）

    Returns:
        JSON 字符串包含文件信息
    """
    # 如果 path 为空或 "."，使用 WORKSPACE.ROOT
    if not path or path == ".":
        path = utils.env("WORKSPACE.ROOT", default=".")

    fp = _resolve_under_root(path)

    if not fp.exists():
        return json.dumps({"ok": False, "error": f"路径不存在: {path}"}, ensure_ascii=False)

    import os
    import datetime

    try:
        stat = fp.stat()
        is_dir = fp.is_dir()
        is_file = fp.is_file()

        info = {
            "path": str(fp.relative_to(utils.get_workspace_root())),
            "name": fp.name,
            "is_dir": is_dir,
            "is_file": is_file,
            "size": stat.st_size if is_file else 0,
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.datetime.fromtimestamp(stat.st_atime).isoformat(),
        }

        if is_dir:
            info["children_count"] = len(list(fp.iterdir()))

        return json.dumps({
            "ok": True,
            "info": info
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("read_file_lines")
def read_file_lines(path: str, start_line: int = 1, end_line: int = 100) -> str:
    """按行读取文件内容。

    Args:
        path: 文件路径（相对于 WORKSPACE_ROOT）
        start_line: 起始行号（从 1 开始）
        end_line: 结束行号

    Returns:
        JSON 字符串包含文件内容和行号信息
    """
    fp = _resolve_under_root(path)

    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    if not fp.is_file():
        return json.dumps({"ok": False, "error": f"不是文件: {path}"}, ensure_ascii=False)

    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()

        # 确保范围有效
        start_line = max(1, start_line)
        end_line = min(len(lines), end_line)

        selected_lines = []
        for i in range(start_line - 1, end_line):
            selected_lines.append({
                "line": i + 1,
                "content": lines[i]
            })

        return json.dumps({
            "ok": True,
            "path": str(fp.relative_to(utils.get_workspace_root())),
            "total_lines": len(lines),
            "start_line": start_line,
            "end_line": end_line,
            "lines": selected_lines
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("walk_dir")
def walk_dir(path: str = ".", max_depth: int = 3, max_files: int = 200) -> str:
    """遍历目录结构。

    Args:
        path: 起始目录路径（"." 表示当前工作目录）
        max_depth: 最大遍历深度，默认 3
        max_files: 最大文件数，默认 200

    Returns:
        JSON 字符串包含目录树结构
    """
    # 使用正确的搜索根目录
    root = _get_search_root(path)
    result = []
    file_count = [0]  # 使用列表以便在嵌套函数中修改

    def walk_recursive(dir_path, current_depth):
        if file_count[0] >= max_files:
            return

        try:
            for item in sorted(dir_path.iterdir()):
                if file_count[0] >= max_files:
                    break

                rel_path = str(item.relative_to(utils.get_workspace_root()))
                is_dir = item.is_dir()

                entry = {
                    "path": rel_path,
                    "name": item.name,
                    "type": "dir" if is_dir else "file",
                    "depth": current_depth
                }

                if is_dir and current_depth < max_depth:
                    result.append(entry)
                    walk_recursive(item, current_depth + 1)
                elif not is_dir:
                    try:
                        entry["size"] = item.stat().st_size
                    except Exception:
                        entry["size"] = 0
                    result.append(entry)
                    file_count[0] += 1
                else:
                    result.append(entry)

        except PermissionError:
            pass

    walk_recursive(root, 0)

    return json.dumps({
        "ok": True,
        "root": str(root.relative_to(utils.get_workspace_root())),
        "max_depth": max_depth,
        "total_items": len(result),
        "items": result
    }, ensure_ascii=False)

@utils.timer
@tool("get_dir_tree")
def get_dir_tree(path: str = ".", max_depth: int = 5) -> str:
    """获取目录树结构（更简洁的树形展示）。

    Args:
        path: 起始目录路径（"." 表示当前工作目录）
        max_depth: 最大深度，默认 5

    Returns:
        JSON 字符串包含树形结构
    """
    # 使用正确的搜索根目录
    root = _get_search_root(path)
    tree = []

    def build_tree(dir_path, prefix: str, is_last: bool, depth: int):
        if depth > max_depth:
            return

        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            total = len(items)

            for i, item in enumerate(items):
                is_last_item = (i == total - 1)
                connector = "└── " if is_last_item else "├── "
                full_prefix = prefix + connector if prefix else ""

                if item.is_dir():
                    tree.append({
                        "prefix": full_prefix,
                        "name": item.name + "/",
                        "type": "dir",
                        "path": str(item.relative_to(utils.get_workspace_root()))
                    })
                    new_prefix = prefix + ("    " if is_last_item else "│   ")
                    build_tree(item, new_prefix, is_last_item, depth + 1)
                else:
                    try:
                        size = item.stat().st_size
                        size_str = f" ({size} bytes)" if size > 0 else ""
                        tree.append({
                            "prefix": full_prefix,
                            "name": item.name + size_str,
                            "type": "file",
                            "path": str(item.relative_to(utils.get_workspace_root()))
                        })
                    except Exception:
                        tree.append({
                            "prefix": full_prefix,
                            "name": item.name,
                            "type": "file",
                            "path": str(item.relative_to(utils.get_workspace_root()))
                        })

        except PermissionError:
            pass

    build_tree(root, "", True, 0)

    return json.dumps({
        "ok": True,
        "root": str(root.relative_to(utils.get_workspace_root())),
        "max_depth": max_depth,
        "total": len(tree),
        "tree": tree
    }, ensure_ascii=False)


# ============================
# C++ Large-Repo Review Skills
# ============================
# These tools are designed for reviewing large C++ repositories with evidence-based navigation and verification loops.
# Conventions:
# - All paths are resolved under WORKSPACE_ROOT via _resolve_under_root()
# - All tool returns are JSON strings with {"ok": bool, ...}

def _truncate_text(s: str, limit: int = 20000) -> str:
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, {len(s)} chars total)"


def _relpath_str(p: Path) -> str:
    try:
        return str(p.relative_to(utils.get_workspace_root()))
    except Exception:
        return str(p)


def _detect_build_system(root: Path) -> str:
    # Simple heuristics: bazel > cmake > make
    if (root / "WORKSPACE").exists() or (root / "WORKSPACE.bazel").exists():
        return "bazel"
    if (root / "CMakeLists.txt").exists() or list(root.rglob("CMakeLists.txt"))[:1]:
        return "cmake"
    if (root / "Makefile").exists():
        return "make"
    return "unknown"


def _find_first(root: Path, names: List[str], max_depth: int = 6) -> Path:
    # breadth-ish: walk limited depth
    try:
        for p in root.rglob("*"):
            if p.name in names:
                # depth limit
                try:
                    if len(p.relative_to(root).parts) <= max_depth:
                        return p
                except Exception:
                    return p
    except Exception:
        pass
    return Path()

@utils.timer
@tool("repo_scan")
def repo_scan(path: str = ".", max_files: int = 20000, max_depth: int = 4) -> str:
    """扫描代码仓库概况（模块边界、语言/文件分布、构建体系、关键文件存在性）。

    Args:
        path: 仓库根目录（相对 WORKSPACE_ROOT）
        max_files: 扫描的最大文件数（防止超大仓库耗时）
        max_depth: 统计 top-level 模块时的最大深度
    """
    root = _get_search_root(path)
    if not root.exists():
        return json.dumps({"ok": False, "error": f"路径不存在: {path}"}, ensure_ascii=False)

    build_system = _detect_build_system(root)

    # Count files by extension (limited)
    exts = {}
    total = 0
    top_dirs = {}
    key_files = {
        "compile_commands": None,
        "cmake_lists": None,
        "bazel_workspace": None,
        "bazel_build": None,
        "clang_tidy": None,
        "clang_format": None,
    }

    # quick key files
    cc = _find_first(root, ["compile_commands.json"], max_depth=8)
    if cc and cc.exists():
        key_files["compile_commands"] = _relpath_str(cc)
    cm = _find_first(root, ["CMakeLists.txt"], max_depth=8)
    if cm and cm.exists():
        key_files["cmake_lists"] = _relpath_str(cm)
    ws = _find_first(root, ["WORKSPACE", "WORKSPACE.bazel"], max_depth=8)
    if ws and ws.exists():
        key_files["bazel_workspace"] = _relpath_str(ws)
    bd = _find_first(root, ["BUILD", "BUILD.bazel"], max_depth=8)
    if bd and bd.exists():
        key_files["bazel_build"] = _relpath_str(bd)
    ct = _find_first(root, [".clang-tidy"], max_depth=8)
    if ct and ct.exists():
        key_files["clang_tidy"] = _relpath_str(ct)
    cf = _find_first(root, [".clang-format"], max_depth=8)
    if cf and cf.exists():
        key_files["clang_format"] = _relpath_str(cf)

    # Walk files
    for p in root.rglob("*"):
        try:
            if total >= max_files:
                break
            if p.is_dir():
                continue
            total += 1
            suf = p.suffix.lower() or "<noext>"
            exts[suf] = exts.get(suf, 0) + 1

            # top-level module bucket by first N parts
            rel = p.relative_to(root)
            parts = rel.parts
            bucket = parts[0] if parts else "."
            top_dirs[bucket] = top_dirs.get(bucket, 0) + 1
        except Exception:
            continue

    # Keep only most common extensions
    common_exts = sorted(exts.items(), key=lambda kv: kv[1], reverse=True)[:30]
    common_dirs = sorted(top_dirs.items(), key=lambda kv: kv[1], reverse=True)[:40]

    return json.dumps({
        "ok": True,
        "root": _relpath_str(root),
        "build_system_guess": build_system,
        "file_count_scanned": total,
        "top_dirs": common_dirs,
        "top_extensions": common_exts,
        "key_files": key_files,
        "notes": [
            "建议：若需要 clangd/clang-tidy 的语义能力，请确保 compile_commands.json 可用（可用 ensure_compile_commands 生成）。",
            "建议：大仓 review 优先用 grep + 符号查询定位证据，再读取局部片段（read_file_lines）。",
        ]
    }, ensure_ascii=False)

@utils.timer
@tool("read_file_lines")
def read_file_lines(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """按行读取文件片段（包含行号），用于证据引用与精确定位。

    Args:
        path: 文件路径（相对 WORKSPACE_ROOT）
        start_line: 起始行（1-based）
        end_line: 结束行（1-based，包含）
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line

    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)
        s = min(start_line, total_lines)
        e = min(end_line, total_lines)
        # slice is 0-based
        chunk = lines[s - 1:e]
        numbered = [{"line": i, "text": chunk[i - s]} for i in range(s, e + 1)]
        return json.dumps({
            "ok": True,
            "path": _relpath_str(fp),
            "start_line": s,
            "end_line": e,
            "total_lines": total_lines,
            "lines": numbered
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("rg_search")
def rg_search(pattern: str, path: str = ".", glob: str = "", max_results: int = 200) -> str:
    """使用 ripgrep (rg) 搜索；若 rg 不可用则回退到 Python 扫描。

    Args:
        pattern: 搜索模式（rg 正则）
        path: 搜索根目录（相对 WORKSPACE_ROOT）
        glob: 文件过滤，例如 "*.cpp" 或 "*.{h,cpp}"
        max_results: 最大返回条数
    """
    root = _get_search_root(path)
    if not root.exists():
        return json.dumps({"ok": False, "error": f"路径不存在: {path}"}, ensure_ascii=False)

    rg = shutil.which("rg")
    results = []

    try:
        if rg:
            cmd = ["rg", "--line-number", "--no-heading", "--smart-case", pattern, str(root)]
            if glob:
                cmd = ["rg", "--line-number", "--no-heading", "--smart-case", "--glob", glob, pattern, str(root)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = proc.stdout.splitlines() if proc.stdout else []
            for line in out[:max_results]:
                # format: file:line:match
                m = re.match(r"^(.*?):(\d+):(.*)$", line)
                if not m:
                    continue
                f, ln, txt = m.group(1), int(m.group(2)), m.group(3)
                try:
                    rel = _relpath_str(Path(f))
                except Exception:
                    rel = f
                results.append({"file": rel, "line": ln, "text": txt})
            return json.dumps({
                "ok": True,
                "engine": "rg",
                "pattern": pattern,
                "root": _relpath_str(root),
                "count": len(results),
                "results": results
            }, ensure_ascii=False)

        # fallback: python scan (slower)
        rx = re.compile(pattern)
        scanned = 0
        for fp in root.rglob("*"):
            if len(results) >= max_results:
                break
            if fp.is_dir():
                continue
            scanned += 1
            try:
                if glob:
                    # basic glob check against name
                    if not fp.match(glob):
                        continue
                txt = fp.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(txt.splitlines(), start=1):
                    if rx.search(line):
                        results.append({"file": _relpath_str(fp), "line": i, "text": line})
                        if len(results) >= max_results:
                            break
            except Exception:
                continue

        return json.dumps({
            "ok": True,
            "engine": "python",
            "pattern": pattern,
            "root": _relpath_str(root),
            "scanned_files": scanned,
            "count": len(results),
            "results": results
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "rg_search 超时", "pattern": pattern}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("ensure_compile_commands")
def ensure_compile_commands(src_dir: str = ".", build_dir: str = "build", method: str = "auto") -> str:
    """确保 compile_commands.json 可用；若缺失则尝试生成（优先 CMake）。

    Args:
        src_dir: 源码根目录（相对 WORKSPACE_ROOT）
        build_dir: 构建目录（相对 WORKSPACE_ROOT）
        method: auto|cmake（目前主支持 cmake；auto 会先探测）
    """
    src = _get_search_root(src_dir)
    if not src.exists():
        return json.dumps({"ok": False, "error": f"src_dir 不存在: {src_dir}"}, ensure_ascii=False)

    # already exists near root or build_dir?
    cc1 = src / "compile_commands.json"
    if cc1.exists():
        return json.dumps({"ok": True, "path": _relpath_str(cc1), "generated": False}, ensure_ascii=False)

    bdir = _resolve_under_root(build_dir)
    cc2 = bdir / "compile_commands.json"
    if cc2.exists():
        return json.dumps({"ok": True, "path": _relpath_str(cc2), "generated": False}, ensure_ascii=False)

    cmake = shutil.which("cmake")
    if method in ("auto", "cmake"):
        # require CMakeLists
        cmakelists = src / "CMakeLists.txt"
        if not cmakelists.exists():
            # maybe it's a subdir project; still allow, but warn
            pass
        if not cmake:
            return json.dumps({"ok": False, "error": "cmake 不可用，无法生成 compile_commands.json"}, ensure_ascii=False)

        try:
            bdir.mkdir(parents=True, exist_ok=True)
            cfg_cmd = [
                "cmake",
                "-S", str(src),
                "-B", str(bdir),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
            cfg = subprocess.run(cfg_cmd, capture_output=True, text=True, timeout=300)
            if cfg.returncode != 0:
                return json.dumps({
                    "ok": False,
                    "error": "cmake configure 失败",
                    "command": " ".join(cfg_cmd),
                    "stdout": _truncate_text(cfg.stdout, 8000),
                    "stderr": _truncate_text(cfg.stderr, 12000),
                }, ensure_ascii=False)

            cc2 = bdir / "compile_commands.json"
            if not cc2.exists():
                return json.dumps({
                    "ok": False,
                    "error": "cmake 已配置但未生成 compile_commands.json（请检查 CMake 版本/生成器）",
                    "build_dir": _relpath_str(bdir),
                }, ensure_ascii=False)

            # Also copy/symlink to src root for convenience
            try:
                if not cc1.exists():
                    # prefer symlink; fallback to copy
                    try:
                        cc1.symlink_to(cc2)
                    except Exception:
                        cc1.write_text(cc2.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            except Exception:
                pass

            return json.dumps({
                "ok": True,
                "path": _relpath_str(cc2),
                "root_link": _relpath_str(cc1) if cc1.exists() else None,
                "generated": True,
                "configure_command": " ".join(cfg_cmd),
            }, ensure_ascii=False)
        except subprocess.TimeoutExpired:
            return json.dumps({"ok": False, "error": "cmake configure 超时"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    return json.dumps({"ok": False, "error": f"不支持的 method: {method}"}, ensure_ascii=False)


def _lsp_send(proc, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    proc.stdin.write(header + body)
    proc.stdin.flush()


def _lsp_read(proc) -> Dict[str, Any]:
    # blocking read; assumes clangd responds quickly
    headers = {}
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("LSP EOF")
        line = line.decode("utf-8", errors="replace").strip()
        if line == "":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    n = int(headers.get("content-length", "0"))
    if n <= 0:
        return {}
    body = proc.stdout.read(n)
    return json.loads(body.decode("utf-8", errors="replace"))


def _file_uri(fp: Path) -> str:
    # file URI for LSP
    import urllib.parse
    return "file://" + urllib.parse.quote(str(fp.resolve()))

@utils.timer
@tool("clangd_lsp_query")
def clangd_lsp_query(path: str, line: int, character: int, method: str = "definition",
                     compile_commands_dir: str = "build", timeout_sec: int = 20) -> str:
    """通过 clangd 的 LSP 接口进行语义查询：definition / references / hover。

    Args:
        path: 源文件路径（相对 WORKSPACE_ROOT）
        line: 行号（1-based）
        character: 列号（1-based，字符偏移）
        method: definition|references|hover
        compile_commands_dir: compile_commands.json 所在目录（相对 WORKSPACE_ROOT）
        timeout_sec: 单次查询超时（秒）
    """
    fp = _resolve_under_root(path)
    if not fp.exists():
        return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)

    clangd = shutil.which("clangd")
    if not clangd:
        return json.dumps({"ok": False, "error": "clangd 不可用，请安装 clangd 或改用 rg_search/grep_cpp 进行定位"},
                          ensure_ascii=False)

    ccdir = _resolve_under_root(compile_commands_dir)
    if not ccdir.exists():
        ccdir = fp.parent  # fallback
    # LSP uses 0-based positions
    pos = {"line": max(0, int(line) - 1), "character": max(0, int(character) - 1)}

    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"ok": False, "error": f"读取文件失败: {e}"}, ensure_ascii=False)

    import time
    start = time.time()

    proc = subprocess.Popen(
        ["clangd", f"--compile-commands-dir={str(ccdir)}", "--background-index=0", "--log=error"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    try:
        # initialize
        init_id = 1
        _lsp_send(proc, {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": _file_uri(utils.get_workspace_root()),
                "capabilities": {},
            }
        })
        # read initialize response
        while True:
            msg = _lsp_read(proc)
            if msg.get("id") == init_id:
                break
            if time.time() - start > timeout_sec:
                raise TimeoutError("clangd initialize timeout")

        _lsp_send(proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}})

        uri = _file_uri(fp)
        _lsp_send(proc, {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {"textDocument": {"uri": uri, "languageId": "cpp", "version": 1, "text": content}}
        })

        req_id = 2
        if method == "definition":
            req = {"jsonrpc": "2.0", "id": req_id, "method": "textDocument/definition",
                   "params": {"textDocument": {"uri": uri}, "position": pos}}
        elif method == "references":
            req = {"jsonrpc": "2.0", "id": req_id, "method": "textDocument/references",
                   "params": {"textDocument": {"uri": uri}, "position": pos, "context": {"includeDeclaration": True}}}
        elif method == "hover":
            req = {"jsonrpc": "2.0", "id": req_id, "method": "textDocument/hover",
                   "params": {"textDocument": {"uri": uri}, "position": pos}}
        else:
            return json.dumps({"ok": False, "error": f"不支持的 method: {method}"}, ensure_ascii=False)

        _lsp_send(proc, req)

        # wait for response id=req_id
        while True:
            if time.time() - start > timeout_sec:
                raise TimeoutError("clangd query timeout")
            msg = _lsp_read(proc)
            if msg.get("id") != req_id:
                continue

            result = msg.get("result")
            if method == "hover":
                return json.dumps({
                    "ok": True,
                    "method": method,
                    "path": _relpath_str(fp),
                    "position": {"line": line, "character": character},
                    "result": result,
                }, ensure_ascii=False)

            # locations
            locs = []
            if isinstance(result, dict) and "uri" in result:
                result = [result]
            if isinstance(result, list):
                for loc in result[:200]:
                    try:
                        u = loc.get("uri", "")
                        rng = loc.get("range", {}) or {}
                        startp = rng.get("start", {}) or {}
                        endp = rng.get("end", {}) or {}
                        # decode file uri
                        if u.startswith("file://"):
                            import urllib.parse
                            fpath = urllib.parse.unquote(u[len("file://"):])
                        else:
                            fpath = u
                        locs.append({
                            "file": _relpath_str(Path(fpath)),
                            "start": {"line": int(startp.get("line", 0)) + 1,
                                      "character": int(startp.get("character", 0)) + 1},
                            "end": {"line": int(endp.get("line", 0)) + 1,
                                    "character": int(endp.get("character", 0)) + 1},
                        })
                    except Exception:
                        continue

            return json.dumps({
                "ok": True,
                "method": method,
                "path": _relpath_str(fp),
                "position": {"line": line, "character": character},
                "count": len(locs),
                "locations": locs,
            }, ensure_ascii=False)

    except TimeoutError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "ok": False,
            "error": str(e),
            "stderr": _truncate_text(proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "", 8000)
        }, ensure_ascii=False)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

@utils.timer
@tool("build_cpp_project")
def build_cpp_project(build_dir: str = "build", build_system: str = "auto", target: str = "", jobs: int = 8,
                      extra_args: str = "", timeout: int = 1800) -> str:
    """构建 C++ 项目（CMake/Make/Bazel），并返回编译输出摘要。

    Args:
        build_dir: 构建目录（相对 WORKSPACE_ROOT）；CMake/ctest 常用
        build_system: auto|cmake|make|bazel
        target: 构建目标（cmake --target / make target / bazel target）
        jobs: 并行度
        extra_args: 额外参数字符串（原样拼接）
        timeout: 超时（秒）
    """
    root = utils.get_workspace_root()
    sys_guess = _detect_build_system(root)
    if build_system == "auto":
        build_system = sys_guess

    try:
        if build_system == "cmake":
            if not shutil.which("cmake"):
                return json.dumps({"ok": False, "error": "cmake 不可用"}, ensure_ascii=False)
            bdir = _resolve_under_root(build_dir)
            if not bdir.exists():
                return json.dumps({"ok": False,
                                   "error": f"build_dir 不存在: {build_dir}（先 ensure_compile_commands 或 cmake configure）"},
                                  ensure_ascii=False)
            cmd = ["cmake", "--build", str(bdir), "-j", str(max(1, jobs))]
            if target:
                cmd += ["--target", target]
            if extra_args:
                cmd += extra_args.split()
        elif build_system == "make":
            if not shutil.which("make"):
                return json.dumps({"ok": False, "error": "make 不可用"}, ensure_ascii=False)
            cmd = ["make", f"-j{max(1, jobs)}"]
            if target:
                cmd.append(target)
            if extra_args:
                cmd += extra_args.split()
        elif build_system == "bazel":
            if not shutil.which("bazel"):
                return json.dumps({"ok": False, "error": "bazel 不可用"}, ensure_ascii=False)
            # default: build all
            bazel_target = target or "//..."
            cmd = ["bazel", "build", bazel_target]
            if extra_args:
                cmd += extra_args.split()
        else:
            return json.dumps({"ok": False, "error": f"未知 build_system: {build_system}"}, ensure_ascii=False)

        res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)
        return json.dumps({
            "ok": res.returncode == 0,
            "build_system": build_system,
            "command": " ".join(cmd),
            "returncode": res.returncode,
            "stdout": _truncate_text(res.stdout or "", 20000),
            "stderr": _truncate_text(res.stderr or "", 20000),
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": f"构建超时 ({timeout}s)"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("run_cpp_tests")
def run_cpp_tests(build_dir: str = "build", test_system: str = "auto", target: str = "", jobs: int = 8,
                  extra_args: str = "", timeout: int = 1800) -> str:
    """运行 C++ 测试（优先 ctest；Bazel 项目则 bazel test）。

    Args:
        build_dir: ctest 的 build 目录
        test_system: auto|ctest|bazel
        target: bazel 测试 target（默认 //...）；ctest 可忽略
        jobs: 并行
        extra_args: 额外参数字符串
        timeout: 超时
    """
    root = utils.get_workspace_root()
    sys_guess = _detect_build_system(root)

    if test_system == "auto":
        test_system = "bazel" if sys_guess == "bazel" else "ctest"

    try:
        if test_system == "ctest":
            if not shutil.which("ctest"):
                return json.dumps({"ok": False, "error": "ctest 不可用"}, ensure_ascii=False)
            bdir = _resolve_under_root(build_dir)
            if not bdir.exists():
                return json.dumps({"ok": False, "error": f"build_dir 不存在: {build_dir}"}, ensure_ascii=False)
            cmd = ["ctest", "--test-dir", str(bdir), "-j", str(max(1, jobs)), "--output-on-failure"]
            if extra_args:
                cmd += extra_args.split()
        elif test_system == "bazel":
            if not shutil.which("bazel"):
                return json.dumps({"ok": False, "error": "bazel 不可用"}, ensure_ascii=False)
            bazel_target = target or "//..."
            cmd = ["bazel", "test", bazel_target]
            if extra_args:
                cmd += extra_args.split()
        else:
            return json.dumps({"ok": False, "error": f"未知 test_system: {test_system}"}, ensure_ascii=False)

        res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=timeout)
        return json.dumps({
            "ok": res.returncode == 0,
            "test_system": test_system,
            "command": " ".join(cmd),
            "returncode": res.returncode,
            "stdout": _truncate_text(res.stdout or "", 20000),
            "stderr": _truncate_text(res.stderr or "", 20000),
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": f"测试超时 ({timeout}s)"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("run_clang_tidy_project")
def run_clang_tidy_project(path: str = ".", build_dir: str = "build", checks: str = "", jobs: int = 8,
                           header_filter: str = "", extra_args: str = "", timeout: int = 1800) -> str:
    """在项目范围运行 clang-tidy（优先 run-clang-tidy，回退到逐文件 clang-tidy）。

    Args:
        path: 目标目录或文件（相对 WORKSPACE_ROOT）
        build_dir: compile_commands.json 所在 build 目录
        checks: clang-tidy -checks=...（可空）
        jobs: 并行
        header_filter: -header-filter=...（可空）
        extra_args: 额外参数，传给 clang-tidy（可空）
        timeout: 超时
    """
    root = _get_search_root(path)
    if not root.exists():
        return json.dumps({"ok": False, "error": f"路径不存在: {path}"}, ensure_ascii=False)

    run_tidy = shutil.which("run-clang-tidy")
    clang_tidy = shutil.which("clang-tidy")
    if not clang_tidy:
        return json.dumps({"ok": False, "error": "clang-tidy 不可用"}, ensure_ascii=False)

    bdir = _resolve_under_root(build_dir)
    if not (bdir / "compile_commands.json").exists():
        return json.dumps(
            {"ok": False, "error": f"{build_dir}/compile_commands.json 不存在；先 ensure_compile_commands"},
            ensure_ascii=False)

    try:
        if run_tidy:
            cmd = ["run-clang-tidy", "-p", str(bdir), "-j", str(max(1, jobs))]
            if checks:
                cmd.append(f"-checks={checks}")
            if header_filter:
                cmd.append(f"-header-filter={header_filter}")
            if extra_args:
                cmd += extra_args.split()
            res = subprocess.run(cmd, cwd=str(utils.get_workspace_root()), capture_output=True, text=True, timeout=timeout)
            return json.dumps({
                "ok": res.returncode == 0,
                "engine": "run-clang-tidy",
                "command": " ".join(cmd),
                "returncode": res.returncode,
                "stdout": _truncate_text(res.stdout or "", 20000),
                "stderr": _truncate_text(res.stderr or "", 20000),
            }, ensure_ascii=False)

        # fallback: per-file
        files = []
        for fp in root.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in [".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"]:
                files.append(fp)
                if len(files) >= 200:  # cap
                    break

        outputs = []
        for fp in files:
            cmd = ["clang-tidy", str(fp), "-p", str(bdir)]
            if checks:
                cmd += [f"-checks={checks}"]
            if header_filter:
                cmd += [f"-header-filter={header_filter}"]
            if extra_args:
                cmd += extra_args.split()
            res = subprocess.run(cmd, cwd=str(utils.get_workspace_root()), capture_output=True, text=True, timeout=120)
            outputs.append({
                "file": _relpath_str(fp),
                "returncode": res.returncode,
                "stdout": _truncate_text(res.stdout or "", 3000),
                "stderr": _truncate_text(res.stderr or "", 3000),
            })

        return json.dumps({
            "ok": True,
            "engine": "clang-tidy-per-file",
            "files_checked": len(files),
            "results": outputs,
            "note": "逐文件模式仅用于兜底；建议安装 run-clang-tidy 提升速度与输出质量。"
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": f"clang-tidy 超时 ({timeout}s)"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("apply_unified_diff")
def apply_unified_diff(diff: str, check_only: bool = False) -> str:
    """应用 unified diff（优先 git apply；否则 fallback patch）。

    Args:
        diff: unified diff 内容
        check_only: 只校验不应用
    """
    if not diff or not diff.strip():
        return json.dumps({"ok": False, "error": "diff 为空"}, ensure_ascii=False)

    git = shutil.which("git")
    patch_bin = shutil.which("patch")

    try:
        if git and (utils.get_workspace_root() / ".git").exists():
            # write temp patch
            tmp = utils.get_workspace_root() / ".tmp_llm_patch.diff"
            tmp.write_text(diff, encoding="utf-8")
            cmd_check = ["git", "apply", "--check", str(tmp)]
            ck = subprocess.run(cmd_check, cwd=str(utils.get_workspace_root()), capture_output=True, text=True, timeout=60)
            if ck.returncode != 0:
                return json.dumps({
                    "ok": False,
                    "error": "git apply --check 失败",
                    "stdout": _truncate_text(ck.stdout or "", 8000),
                    "stderr": _truncate_text(ck.stderr or "", 8000),
                }, ensure_ascii=False)

            if check_only:
                return json.dumps({"ok": True, "checked": True, "applied": False}, ensure_ascii=False)

            cmd_apply = ["git", "apply", str(tmp)]
            ap = subprocess.run(cmd_apply, cwd=str(utils.get_workspace_root()), capture_output=True, text=True, timeout=60)
            if ap.returncode != 0:
                return json.dumps({
                    "ok": False,
                    "error": "git apply 失败",
                    "stdout": _truncate_text(ap.stdout or "", 8000),
                    "stderr": _truncate_text(ap.stderr or "", 8000),
                }, ensure_ascii=False)

            return json.dumps({"ok": True, "checked": True, "applied": True, "engine": "git"}, ensure_ascii=False)

        if not patch_bin:
            return json.dumps({"ok": False, "error": "无 git 仓库且 patch 不可用，无法应用 diff"}, ensure_ascii=False)

        # patch fallback
        tmp = utils.get_workspace_root() / ".tmp_llm_patch.diff"
        tmp.write_text(diff, encoding="utf-8")
        cmd = ["patch", "-p0", "-i", str(tmp)]
        if check_only:
            cmd.insert(1, "--dry-run")
        res = subprocess.run(cmd, cwd=str(utils.get_workspace_root()), capture_output=True, text=True, timeout=60)
        return json.dumps({
            "ok": res.returncode == 0,
            "engine": "patch",
            "checked": True,
            "applied": (not check_only) and res.returncode == 0,
            "stdout": _truncate_text(res.stdout or "", 8000),
            "stderr": _truncate_text(res.stderr or "", 8000),
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({"ok": False, "error": "apply_unified_diff 超时"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

@utils.timer
@tool("patch_apply_and_verify")
def patch_apply_and_verify(diff: str, format_changed: bool = True, build: bool = False, test: bool = False,
                           build_dir: str = "build") -> str:
    """应用 diff 后可选执行：clang-format（仅对改动文件）+ build + test，形成可验证闭环。

    Args:
        diff: unified diff
        format_changed: 是否对改动文件运行 clang-format（需要 .clang-format 或 clang-format 可用）
        build: 是否构建
        test: 是否跑测试
        build_dir: 构建目录
    """
    # parse changed files from diff
    changed = []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            if p == "/dev/null":
                continue
            changed.append(p)
    changed = list(dict.fromkeys(changed))[:50]

    ck = json.loads(apply_unified_diff(diff, check_only=False))
    if not ck.get("ok"):
        return json.dumps({"ok": False, "stage": "apply", "detail": ck}, ensure_ascii=False)

    fmt_results = []
    if format_changed and shutil.which("clang-format"):
        for f in changed:
            # only format cpp-like files
            if Path(f).suffix.lower() not in [".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"]:
                continue
            try:
                res = subprocess.run(["clang-format", "-i", str(_resolve_under_root(f))],
                                     capture_output=True, text=True, timeout=60)
                fmt_results.append(
                    {"file": f, "ok": res.returncode == 0, "stderr": _truncate_text(res.stderr or "", 2000)})
            except Exception as e:
                fmt_results.append({"file": f, "ok": False, "error": str(e)})

    build_res = None
    test_res = None
    if build:
        build_res = json.loads(build_cpp_project(build_dir=build_dir))
    if test:
        test_res = json.loads(run_cpp_tests(build_dir=build_dir))

    return json.dumps({
        "ok": True,
        "changed_files": changed,
        "format_results": fmt_results,
        "build": build_res,
        "test": test_res
    }, ensure_ascii=False)

TOOLS = [
    # 基础工具
    read_file,
    list_dir,
    grep_text,
    web_search,
    web_open,
    http_get,
    ensure_dir,
    write_text_file,
    run_safe_command,
    analyze_code,
    run_tests,
    format_code,
    check_syntax,
    extract_functions,
    get_code_metrics,
    save_mermaid_diagram,
    generate_diagram_description,
    create_plotly_chart,
    save_chart_data,
    analyze_data_for_chart,
    # Bash 和目录工具
    run_bash,
    get_file_info,
    read_file_lines,
    walk_dir,
    get_dir_tree,
    # 通用源码工具
    read_source_file,
    list_source_files,
    # batch_read_source_files,
    grep_source_code,
    get_source_metrics,
    # C/C++ 专用工具
    extract_cpp_functions,
    run_cpp_linter,
    check_cpp_syntax,
    format_cpp,
    find_cmake_files,
    read_compile_commands,
    repo_scan,
    read_file_lines,
    rg_search,
    ensure_compile_commands,
    clangd_lsp_query,
    build_cpp_project,
    run_cpp_tests,
    run_clang_tidy_project,
    apply_unified_diff,
    patch_apply_and_verify,
    # git 工具
    git_status,
    git_diff,
    git_show,
]
TOOL_REGISTRY = build_tool_registry(TOOLS)

# skills_registry.py
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from langchain_core.tools import tool

import utils
from const import TRUNCATE_LOG_LEN, DEFAULT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillMeta:
    name: str
    path: Path
    one_liner: str


class SkillRegistry:
    """
    扫描 ./skills/<name>/SKILL.md
    - 启动时 scan() 生成技能目录（name + one_liner）
    - 运行时 load(name) 按需加载全文
    """

    def __init__(self, skills_dir: Path, *, max_chars: int = 50_000):
        self.skills_dir = skills_dir.resolve()
        self.max_chars = max_chars
        self._skills: Dict[str, SkillMeta] = {}
        self._cache: Dict[str, str] = {}
        self._last_scan_ts = 0.0
        self._last_dir_mtime = 0.0

    def _dir_mtime(self) -> float:
        try:
            return self.skills_dir.stat().st_mtime
        except Exception:
            return 0.0

    def maybe_rescan(self, ttl_s: float = 2.0) -> None:
        now = time.time()
        if now - self._last_scan_ts < ttl_s:
            return
        self._last_scan_ts = now

        cur_mtime = self._dir_mtime()
        if cur_mtime != self._last_dir_mtime:
            self._last_dir_mtime = cur_mtime
            self.scan()

    def scan(self) -> Dict[str, SkillMeta]:
        self._skills.clear()
        self._cache.clear()

        if not self.skills_dir.exists():
            return self._skills

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            fp = skill_dir / "SKILL.md"
            if not fp.exists():
                continue

            name = skill_dir.name
            text = fp.read_text(encoding="utf-8", errors="ignore")
            one_liner = self._extract_one_liner(text) or "（无简介）"
            self._skills[name] = SkillMeta(name=name, path=fp, one_liner=one_liner)

        return self._skills

    def catalog_text(self) -> str:
        """
        给模型看的技能目录（短文本）
        """
        self.maybe_rescan()

        if not self._skills:
            return "（当前无可用 skills）"

        lines = ["可用技能目录（需要时调用 load_skill(name) 读取全文）："]
        for name, meta in sorted(self._skills.items(), key=lambda x: x[0]):
            lines.append(f"- {name}: {meta.one_liner}")
        return "\n".join(lines)

    def load(self, name: str) -> str:
        """
        按需加载某个 skill 的 SKILL.md 全文（可缓存 + 截断）
        """
        self.maybe_rescan()

        name = (name or "").strip()
        if not name:
            raise ValueError("skill name is empty")

        if name in self._cache:
            return self._cache[name]

        meta = self._skills.get(name)
        if not meta:
            raise ValueError(
                f"unknown skill: {name}. available={list(self._skills.keys())}"
            )

        text = meta.path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n...[TRUNCATED]..."

        self._cache[name] = text
        return text

    @staticmethod
    def _extract_one_liner(text: str) -> Optional[str]:
        """
        优先提取 '## 一句话简介' 下第一行非空文本
        否则尝试从 '# SKILL:' 标题下一行提取
        """
        # 1) ## 一句话简介
        m = re.search(r"^##\s*一句话简介\s*$([\s\S]*?)(^\s*---|\Z)", text, flags=re.M)
        if m:
            block = m.group(1)
            for line in block.splitlines():
                line = line.strip(" \t-#")
                if line:
                    return line

        # 2) fallback: 第一段非空
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line

        return None


def _safe_skill_name(name: str) -> str:
    # 只允许 folder 名（防止 ../ 目录穿越）
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name or ""):
        raise ValueError("invalid skill name (allowed: letters/digits/_/-)")
    return name


def _read_text(p: Path, max_chars: int = 20000) -> str:
    return p.read_text(encoding="utf-8", errors="replace")[:max_chars]


def _extract_title_and_summary(md: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in (md or "").splitlines()]
    title = ""
    summary = ""

    # title: 第一行 # xxx
    for ln in lines:
        if ln.startswith("#"):
            title = ln.lstrip("#").strip()
            break

    # summary: 找第一行非空且不是标题的句子
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        summary = ln
        break

    if not title:
        title = "Untitled"
    if not summary:
        summary = ""
    return title, summary


WORKSPACE_ROOT = (
    Path(utils.env("WORKSPACE.ROOT", str(DEFAULT_ROOT))).expanduser().resolve()
)
SKILLS_DIR = (
    Path(utils.env("SKILLS.DIR", str(WORKSPACE_ROOT / "skills"))).expanduser().resolve()
)
# Backward-compat alias (older code uses SKILLS_ROOT)
SKILLS_ROOT = SKILLS_DIR

skill_registry = SkillRegistry(SKILLS_DIR, max_chars=50_000)
skill_registry.scan()

_SKILLS_MARK = "\n# === skills catalog (auto) ===\n"

def with_skill_catalog(system_prompt: str, *, role: str) -> str:
    role2skills = {
        "researcher": ["web_search", "web_open"],
        "solver": ["read_text_file", "run_command", "http_get"],
        "writer": ["write_file", "render_markdown"],
    }
    skills = role2skills.get(role, [])
    if not skills:
        return system_prompt
    catalog = "【可用技能】\n" + "\n".join(f"- {s}" for s in skills)
    return system_prompt + "\n\n" + catalog


# def scan_skills(max_items: int = 200) -> dict:
#     """
#     扫描 skills/<name>/SKILL.md，返回：
#     {
#       "index": {name: {"title":..., "summary":...}},
#       "catalog_text": "...(适合塞 prompt 的目录文本)"
#     }
#     """
#     index = {}
#     if not SKILLS_ROOT.exists():
#
#         logger.error("SKILLS_ROOT does not exist")
#
#         return {"index": {}, "catalog_text": "No skills folder found."}
#
#     cnt = 0
#     for d in sorted(SKILLS_ROOT.iterdir(), key=lambda x: x.name):
#         if cnt >= max_items:
#             break
#         if not d.is_dir():
#             continue
#
#         skill_name = d.name
#         md_path = d / "SKILL.md"
#         if not md_path.exists():
#             continue
#
#         md = _read_text(md_path, max_chars=8000)
#         title, summary = _extract_title_and_summary(md)
#         index[skill_name] = {"title": title, "summary": summary}
#         cnt += 1
#
#     # 生成精简目录（避免太长）
#     lines = ["可用技能目录（按需用 load_skill(name) 加载详情）："]
#     for name, meta in index.items():
#         t = meta.get("title", "")
#         s = meta.get("summary", "")
#         if s:
#             lines.append(f"- {name}: {t} — {s}")
#         else:
#             lines.append(f"- {name}: {t}")
#     catalog_text = "\n".join(lines)
#
#     # 防止 prompt 过长：截断一下
#     if len(catalog_text) > TRUNCATE_LOG_LEN:
#         catalog_text = (
#             catalog_text[:TRUNCATE_LOG_LEN]
#             + "\n...(truncated, use list_skills/load_skill)"
#         )
#         logger.warning(
#             f"WARNING: Skill catalog truncated from {len(catalog_text)} to {TRUNCATE_LOG_LEN} chars"
#         )
#
#     logger.debug("\n".join(lines))
#
#     return {"index": index, "catalog_text": catalog_text}

def _skills_list_impl(max_items: int = 200) -> dict:
    # 触发自动刷新
    skill_registry.maybe_rescan(ttl_s=0.0)  # 或 skill_registry.scan() 也行
    skills = []
    for i, (name, meta) in enumerate(sorted(skill_registry._skills.items(), key=lambda x: x[0])):
        if i >= max_items:
            break
        skills.append({"name": name, "one_liner": meta.one_liner})
    return {"ok": True, "count": len(skills), "skills": skills, "catalog_text": skill_registry.catalog_text()}

@tool("skills_list")
def skills_list(max_items: int = 200) -> dict:
    """
    列出当前可用的 skills（技能文档）目录与摘要信息。

    说明：
    - 本工具只用于“发现技能/查看目录”，不会执行任何技能步骤。
    - 返回内容包含：skills 列表（name/one_liner）以及可用于提示词注入的 catalog_text。
    - 如需查看某个技能的完整说明，请调用 skills_load(name)。

    Args:
        max_items: 最多返回多少条技能目录项（用于控制输出大小/token）。

    Returns:
        dict: {
          "ok": bool,
          "count": int,
          "skills": [{"name": str, "one_liner": str}, ...],
          "catalog_text": str
        }
    """
    return _skills_list_impl(max_items=max_items)

@tool("skills_load")
def skills_load(name: str) -> str:
    """
    加载指定 skill 的 SKILL.md 全文（技能说明/执行步骤/注意事项）。

    说明：
    - 本工具只“读取技能文档内容”，不会执行技能中提到的任何命令或工具流程。
    - name 必须是安全的目录名（由 _safe_skill_name 校验），防止路径穿越。
    - 当 skills 目录发生变更时，会先触发一次 rescan（ttl_s=0.0）。

    Args:
        name: 技能名（对应 skills/<name>/SKILL.md）。

    Returns:
        str: 技能文档全文（可能按 max_chars 截断，取决于 skill_registry 的配置）。
    """
    skill_registry.maybe_rescan(ttl_s=0.0)
    name = _safe_skill_name((name or "").strip())
    return skill_registry.load(name)

@tool("skills_rescan")
def skills_rescan(max_items: int = 200) -> dict:
    """
    重新扫描 skills 目录并刷新缓存，然后返回最新的 skills 目录。

    说明：
    - 用于显式刷新（通常不必频繁调用；skills_list/skills_load 已可触发自动刷新）。
    - 本工具同样不会执行技能步骤，仅更新技能索引与缓存。

    Args:
        max_items: 最多返回多少条技能目录项。

    Returns:
        dict: 同 skills_list 的返回结构。
    """
    skill_registry.scan()
    return _skills_list_impl(max_items=max_items)
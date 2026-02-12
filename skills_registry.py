# skills_registry.py
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import yaml
from langchain_core.tools import tool

import utils
from const import TRUNCATE_LOG_LEN

logger = logging.getLogger(__name__)

_MD_CANDIDATES = ("SKILL.md", "skill.md", "README.md")

@dataclass
class SkillMeta:
    """Best-effort parsed metadata for a skill.

    Notes:
    - No rigid file format is required. Metadata extraction is best-effort.
    - If allowed_tools is empty, it means "unknown/unspecified" (not "no tools").
    """

    skill_id: str
    source: str  # e.g. 'local', 'external'
    path: Optional[Path] = None

    display_name: str = ""
    one_liner: str = ""
    description: str = ""
    triggers: List[str] = field(default_factory=list)

    allowed_tools: List[str] = field(default_factory=list)
    allowed_tools_explicit: bool = False  # whether allowed_tools was explicitly found

    doc_chars: int = 0

def _safe_read_text(p: Path, max_chars: int = 400_000) -> str:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        return txt[:max_chars]
    except Exception as e:
        logger.warning("failed to read %s: %s", str(p), e)
        return ""


def _split_front_matter(md: str) -> Tuple[Optional[str], str]:
    """Return (front_matter_yaml_text_or_none, body)."""
    s = md.lstrip()
    if not s.startswith("---"):
        return None, md
    # Find closing '---' on its own line.
    m = re.search(r"\n---\s*\n", s)
    if not m:
        return None, md
    start = s.find("---") + 3
    front = s[start:m.start()].strip("\r\n ")
    body = s[m.end():]
    return front, body

def _yaml_to_dict(front: str) -> Dict[str, Any]:
    if not front.strip():
        return {}
    if yaml is None:
        out: Dict[str, Any] = {}
        for line in front.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
        return out
    try:
        obj = yaml.safe_load(front)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _norm_key(k: str) -> str:
    return re.sub(r"[_\s]+", "-", (k or "").strip().lower())

def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    parts = re.split(r"[\s,;]+", s)
    return [p for p in (x.strip() for x in parts) if p]

def _extract_heading_name(md: str) -> str:
    # '# SKILL: xxx' or '# Skill: xxx'
    m = re.search(r"^#\s*SKILL\s*[:：]\s*(.+?)\s*$", md, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    # YAML-like 'name: xxx' near the top (first 60 lines)
    head = "\n".join(md.splitlines()[:60])
    m = re.search(r"^name\s*:\s*(.+?)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    # First H1
    m = re.search(r"^#\s+(.+?)\s*$", md, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""

def _extract_one_liner(md: str) -> str:
    m = re.search(
        r"^##\s+One-Sentence\s+Summary\s*\n(.+?)(?:\n\n|\n#|\n##)",
        md,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if m:
        line = m.group(1).strip().splitlines()[0].strip()
        return line[:300]
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    if not lines:
        return ""
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    if not lines:
        return ""
    return lines[0][:300]

def _parse_meta(md: str, fallback_id: str) -> SkillMeta:
    front, _ = _split_front_matter(md)
    front_obj: Dict[str, Any] = _yaml_to_dict(front or "")

    norm: Dict[str, Any] = {}
    for k, v in front_obj.items():
        norm[_norm_key(str(k))] = v

    name = str(norm.get("name") or norm.get("skill") or "").strip()
    if not name:
        name = _extract_heading_name(md).strip()

    description = str(norm.get("description") or "").strip()
    one_liner = str(norm.get("one-liner") or norm.get("one-liner-summary") or "").strip()
    if not one_liner:
        one_liner = _extract_one_liner(md)

    triggers = _as_list(norm.get("triggers") or norm.get("trigger") or norm.get("keywords"))
    if not triggers:
        head = "\n".join(md.splitlines()[:120])
        m = re.search(r"^Triggers\s*:\s*(.+?)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            triggers = _as_list(m.group(1))

    allowed_raw = norm.get("allowed-tools") or norm.get("allowed_tools") or norm.get("allowedtool") or norm.get("allowed")
    allowed_tools = _as_list(allowed_raw)
    allowed_explicit = bool(allowed_tools)
    if not allowed_tools:
        head = "\n".join(md.splitlines()[:120])
        m = re.search(r"^allowed[-_ ]?tools\s*:\s*(.+?)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            allowed_tools = _as_list(m.group(1))
            allowed_explicit = bool(allowed_tools)

    return SkillMeta(
        skill_id=fallback_id,
        source="local",
        display_name=name or fallback_id,
        one_liner=one_liner,
        description=description,
        triggers=triggers,
        allowed_tools=allowed_tools,
        allowed_tools_explicit=allowed_explicit,
        doc_chars=len(md),
    )

def _pick_doc_path(skill_dir: Path) -> Optional[Path]:
    for fn in _MD_CANDIDATES:
        p = skill_dir / fn
        if p.exists() and p.is_file():
            return p
    mds = sorted([p for p in skill_dir.glob("*.md") if p.is_file()])
    return mds[0] if mds else None


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillMeta] = {}
        self._cache_text: Dict[str, str] = {}

    @utils.timer
    def scan_skills(self) -> int:
        self._skills.clear()
        self._cache_text.clear()

        skills_dir = utils.get_skills_dir()
        for d in sorted([p for p in skills_dir.iterdir() if p.is_dir()]):
            doc_path = _pick_doc_path(d)
            if not doc_path:
                continue
            md = _safe_read_text(doc_path)
            if not md.strip():
                continue

            skill_id = d.name
            meta = _parse_meta(md, fallback_id=skill_id)
            meta.path = doc_path
            meta.source = "local"

            self._skills[skill_id] = meta
            self._cache_text[skill_id] = md

        return len(self._skills)

    @utils.timer
    def list_skills(self, max_items: int = 200) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for k in sorted(self._skills.keys()):
            meta = self._skills[k]
            out.append(
                {
                    "id": meta.skill_id,
                    "name": meta.display_name,
                    "one_liner": meta.one_liner,
                    "triggers": meta.triggers,
                    "allowed_tools": meta.allowed_tools,
                    "allowed_tools_explicit": meta.allowed_tools_explicit,
                    "source": meta.source,
                    "path": str(meta.path) if meta.path else None,
                }
            )
            if len(out) >= max_items:
                break
        return out

    @utils.timer
    def load_skill(self, skill_id: str, max_chars: int = 60_000) -> str:
        skill_id = (skill_id or "").strip()
        if not skill_id:
            return ""
        txt = self._cache_text.get(skill_id)
        if txt is None and skill_id in self._skills and self._skills[skill_id].path:
            txt = _safe_read_text(self._skills[skill_id].path or Path("/dev/null"))
            self._cache_text[skill_id] = txt
        if txt is None:
            sid = self._match_by_name(skill_id)
            if sid:
                return self.load_skill(sid, max_chars=max_chars)
            return f"[skills_load] skill not found: {skill_id}"
        return txt[:max_chars]

    def _match_by_name(self, name: str) -> Optional[str]:
        needle = (name or "").strip().lower()
        if not needle:
            return None
        for sid, meta in self._skills.items():
            if meta.display_name.strip().lower() == needle:
                return sid
        return None

    @utils.timer
    def register_external(self, skill_id: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        skill_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", (skill_id or "").strip())
        if not skill_id:
            return {"ok": False, "error": "empty skill_id"}

        target_dir = utils.get_skills_dir() / "_external" / skill_id
        target_path = target_dir / "SKILL.md"
        if target_path.exists() and not overwrite:
            return {"ok": False, "error": "skill already exists", "path": str(target_path)}

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content or "", encoding="utf-8")
        self.scan_skills()
        return {"ok": True, "id": skill_id, "path": str(target_path)}

    def _visible_for_toolset(self, meta: SkillMeta, toolset: set[str]) -> bool:
        if not meta.allowed_tools:
            return True
        return set(meta.allowed_tools).issubset(toolset)

    @utils.timer
    def search_cards(self, query: str, toolset: set[str], top_k: int = 8) -> List[SkillMeta]:
        q = (query or "").strip().lower()
        if not q:
            return []

        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", q, flags=re.IGNORECASE)
        if not tokens:
            tokens = [q]

        scored: List[Tuple[float, SkillMeta]] = []
        for meta in self._skills.values():
            if not self._visible_for_toolset(meta, toolset):
                continue
            card_text = " ".join(
                [
                    meta.display_name or "",
                    meta.one_liner or "",
                    meta.description or "",
                    " ".join(meta.triggers or []),
                    " ".join(meta.allowed_tools or []),
                ]
            ).lower()

            score = 0.0
            for t in meta.triggers:
                tl = (t or "").strip().lower()
                if tl and tl in q:
                    score += 5.0

            for tok in tokens:
                if tok in card_text:
                    score += 1.0 + min(len(tok) / 8.0, 1.5)

            if meta.display_name and meta.display_name.lower() in q:
                score += 2.0

            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[: max(1, int(top_k))]]

    @utils.timer
    def render_cards(self, metas: List[SkillMeta], max_items: int = 8) -> str:
        metas = metas[:max_items]
        if not metas:
            return ""
        lines: List[str] = []
        lines.append("【Skill Cards（仅摘要，必要时用 skills_load 打开全文）】")
        for i, m in enumerate(metas, 1):
            tools = " ".join(m.allowed_tools) if m.allowed_tools else "(unspecified)"
            trig = ", ".join(m.triggers[:12]) + ("..." if len(m.triggers) > 12 else "")
            lines.append(
                f"{i}. id={m.skill_id} | name={m.display_name}\n"
                f"   one_liner={m.one_liner}\n"
                f"   allowed_tools={tools}\n"
                f"   triggers={trig}"
            )
        lines.append("\n选择规则：只选与你当前任务相关的少量技能；如需细节，调用 skills_load(id)。")
        return "\n".join(lines)

    @utils.timer
    def render_full_docs(self, skill_ids: List[str], max_chars_total: int = 12_000) -> str:
        if not skill_ids:
            return ""
        out: List[str] = ["【Selected Skills（全文/片段）】"]
        remaining = max_chars_total
        for sid in skill_ids:
            doc = self.load_skill(sid, max_chars=remaining)
            if not doc:
                continue
            if len(doc) > remaining:
                doc = doc[:remaining]
            out.append(f"\n===== SKILL: {sid} =====\n" + doc)
            remaining -= len(doc)
            if remaining <= 500:
                break
        return "\n".join(out)


REGISTRY = SkillRegistry()
REGISTRY.scan_skills()

@utils.timer
@tool("skills_rescan")
def skills_rescan() -> str:
    """Rescan skills directory and refresh cache."""
    t0 = time.time()
    n = REGISTRY.scan_skills()
    return f"skills_rescan ok: {n} skills, cost={time.time()-t0:.3f}s"

@utils.timer
@tool("skills_list")
def skills_list(max_items: int = 200) -> str:
    """List skills (metadata only)."""
    items = REGISTRY.list_skills(max_items=int(max_items))
    return json.dumps(items, ensure_ascii=False, indent=2, default=str)[:TRUNCATE_LOG_LEN]

@utils.timer
@tool("skills_load")
def skills_load(skill_id: str, max_chars: int = 60_000) -> str:
    """Load a skill markdown by id (folder name) or display name."""
    return REGISTRY.load_skill(skill_id, max_chars=int(max_chars))[:TRUNCATE_LOG_LEN]

@utils.timer
@tool("skills_register")
def skills_register(skill_id: str, content: str, overwrite: bool = False) -> str:
    """Register a skill from raw markdown content (e.g., fetched online).

    The skill is written under: skills/_external/<skill_id>/SKILL.md.
    Then the registry is rescanned.
    """
    res = REGISTRY.register_external(skill_id, content, overwrite=bool(overwrite))
    return json.dumps(res, ensure_ascii=False, indent=2, default=str)[:TRUNCATE_LOG_LEN]

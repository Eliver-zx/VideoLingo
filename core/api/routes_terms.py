from __future__ import annotations

import json
from pathlib import Path
import re
import threading
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from core.api.helpers import _resolve_workspace
from core.api.schemas import DomainTermSyncRequest, TermsUpdateRequest
from core.utils.config_utils import load_key_or
from core.workspace import (
    DEFAULT_GLOSSARY_DOMAIN,
    TaskWorkspace,
    domain_csv_known_srcs,
    domain_dict_json_path,
    promote_terms_to_domain,
    remove_terms_from_domain,
)
from core.workspace.fs_atomic import register_cleanup_hook

router = APIRouter()

_NEWLINE_STRIP_RE = re.compile(r"\s*\n\s*")
_COLON_SPLIT_RE = re.compile(r"[：:]")
_DICT_META_KEYS = frozenset(("version", "language", "type", "id", "name"))

_DOMAIN_DICT_SRCS_CACHE: dict[Path, tuple[tuple[int, int, int], frozenset[str]]] = {}
_DICT_SRCS_LOCK = threading.Lock()


def _purge_domain_dict_srcs_cache(target_resolved: Path) -> None:
    with _DICT_SRCS_LOCK:
        for k in [
            p
            for p in _DOMAIN_DICT_SRCS_CACHE
            if p == target_resolved or target_resolved in p.parents
        ]:
            _DOMAIN_DICT_SRCS_CACHE.pop(k, None)


register_cleanup_hook(_purge_domain_dict_srcs_cache)


def _clean_tgt_str(val: str) -> str:
    return (_NEWLINE_STRIP_RE.sub("", val) if "\n" in val else val).strip()


def _collect_raw_dict_pairs(node: Any, pairs: dict[str, tuple[str, str]]) -> None:
    if isinstance(node, dict):
        src_val = (
            node.get("src")
            or node.get("en")
            or node.get("source")
            or node.get("term")
            or node.get("original")
        )
        tgt_val = (
            node.get("tgt")
            or node.get("zh-chs")
            or node.get("zh-cn")
            or node.get("zh")
            or node.get("target")
            or node.get("translation")
        )
        if isinstance(src_val, str) and isinstance(tgt_val, str):
            s = src_val.strip()
            t = _clean_tgt_str(tgt_val)
            if s and t and s.lower() != t.lower():
                key = s.lower()
                if key not in pairs:
                    pairs[key] = (s, t)
                return
        for k, v in node.items():
            if isinstance(v, str):
                s = str(k).strip()
                t = _clean_tgt_str(v)
                if (
                    s
                    and t
                    and len(s) <= 80
                    and len(t) <= 80
                    and s.lower() != t.lower()
                    and s.lower() not in _DICT_META_KEYS
                ):
                    key = s.lower()
                    if key not in pairs:
                        pairs[key] = (s, t)
            elif isinstance(v, (dict, list)):
                _collect_raw_dict_pairs(v, pairs)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _collect_raw_dict_pairs(item, pairs)


def _extract_dict_src_keys(raw_dom: Any) -> frozenset[str]:
    pairs: dict[str, tuple[str, str]] = {}
    _collect_raw_dict_pairs(raw_dom, pairs)
    keys = set(pairs.keys())
    for s, t in pairs.values():
        if ":" in s and ("：" in t or ":" in t):
            s_parts = [p.strip() for p in s.split(":") if p.strip()]
            t_parts = [p.strip() for p in _COLON_SPLIT_RE.split(t) if p.strip()]
            if len(s_parts) == 2 and len(t_parts) == 2:
                tail_low = s_parts[1].lower()
                if tail_low:
                    keys.add(tail_low)
    return frozenset(keys)


def _resolve_task_domain(ws: TaskWorkspace) -> str:
    manifest = ws.load_manifest()
    overrides = dict(manifest.get("config_override") or {})
    task_domain = overrides.get("glossary_domain")
    if task_domain is None:
        return str(
            load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
        ).strip()
    return str(task_domain).strip()


def _domain_known_src_sets(domain: str) -> tuple[frozenset[str], frozenset[str]]:
    if not domain:
        return frozenset(), frozenset()
    try:
        csv_srcs = domain_csv_known_srcs(domain)
    except ValueError:
        return frozenset(), frozenset()
    try:
        dict_path = domain_dict_json_path(domain)
    except ValueError:
        return csv_srcs, frozenset()
    try:
        st = dict_path.stat()
    except OSError:
        return csv_srcs, frozenset()
    resolved = dict_path.resolve()
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _DICT_SRCS_LOCK:
        cached = _DOMAIN_DICT_SRCS_CACHE.get(resolved)
        if cached is not None and cached[0] == sig:
            return csv_srcs, cached[1]

    try:
        raw_dom = json.loads(dict_path.read_text(encoding="utf-8-sig"))
        dict_srcs = _extract_dict_src_keys(raw_dom)
        with _DICT_SRCS_LOCK:
            _DOMAIN_DICT_SRCS_CACHE[resolved] = (sig, dict_srcs)
        return csv_srcs, dict_srcs
    except (OSError, ValueError):
        return csv_srcs, frozenset()


def _load_enriched_task_terms(ws: TaskWorkspace) -> dict[str, Any]:
    if not ws.has_terms():
        return {"summary": "", "terms": []}
    data = ws.load_terms()
    task_domain = _resolve_task_domain(ws)
    csv_srcs, dict_srcs = _domain_known_src_sets(task_domain)
    promoted_here = {
        str(s).strip().lower()
        for s in (ws.get_kv("task_promoted_srcs", []) or [])
        if str(s).strip()
    }
    enriched_terms: list[dict[str, Any]] = []
    for item in data.get("terms") or []:
        k = str(item.get("src") or "").strip().lower()
        src_tag = str(item.get("source") or "task")
        in_csv = bool(k and k in csv_srcs)
        if k in promoted_here or src_tag == "manual":
            in_domain_kb = False
        elif (
            src_tag in ("domain_kb", "global_kb", "domain_dict", "custom", "json_kb")
            or in_csv
            or (k in dict_srcs)
        ):
            in_domain_kb = True
        else:
            in_domain_kb = False
        enriched_terms.append(
            {
                **item,
                "in_domain_kb": in_domain_kb,
                "promoted": in_csv,
            }
        )
    return {"summary": data.get("summary", ""), "terms": enriched_terms}


@router.get("/api/tasks/{task_id}/terms")
def get_task_terms(task_id: str):
    ws = _resolve_workspace(task_id)
    payload = _load_enriched_task_terms(ws)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return Response(content=body, media_type="application/json")


@router.post("/api/tasks/{task_id}/terms/domain")
def sync_task_terms_with_domain(task_id: str, request: DomainTermSyncRequest):
    ws = _resolve_workspace(task_id)
    task_domain = _resolve_task_domain(ws)
    if not task_domain:
        raise HTTPException(
            422, "Task has no bound glossary domain; select a domain first."
        )
    promoted_here = {
        str(s).strip().lower()
        for s in (ws.get_kv("task_promoted_srcs", []) or [])
        if str(s).strip()
    }
    if request.action == "add":
        items = [
            t
            for t in (request.terms or [])
            if str(t.get("src") or "").strip() and str(t.get("tgt") or "").strip()
        ]
        if not items:
            raise HTTPException(422, "Term `src` and `tgt` must be non-empty.")
        csv_before, dict_srcs = _domain_known_src_sets(task_domain)
        count = promote_terms_to_domain(items, domain=task_domain)
        for t in items:
            k = str(t["src"]).strip().lower()
            if k not in csv_before and k not in dict_srcs:
                promoted_here.add(k)
        ws.set_kv("task_promoted_srcs", sorted(promoted_here))
        return {"promoted": count, "removed": 0, "glossary_domain": task_domain}

    srcs = [str(s or "").strip() for s in (request.srcs or []) if str(s or "").strip()]
    if not srcs and request.terms:
        srcs = [
            str(t.get("src") or "").strip()
            for t in request.terms
            if str(t.get("src") or "").strip()
        ]
    removed = remove_terms_from_domain(srcs, domain=task_domain)
    for s in srcs:
        promoted_here.discard(s.lower())
    ws.set_kv("task_promoted_srcs", sorted(promoted_here))
    return {"promoted": 0, "removed": removed, "glossary_domain": task_domain}


@router.put("/api/tasks/{task_id}/terms")
def update_task_terms(task_id: str, request: TermsUpdateRequest):
    ws = _resolve_workspace(task_id)
    manifest = ws.load_manifest()
    overrides = dict(manifest.get("config_override") or {})
    if request.glossary_domain is not None:
        overrides["glossary_domain"] = request.glossary_domain.strip()
        manifest["config_override"] = overrides
        ws.save_manifest(manifest)

    task_domain = _resolve_task_domain(ws)

    if request.terms is None:
        current = _load_enriched_task_terms(ws)
        return {
            "saved": len(current.get("terms") or []),
            "promoted_to_domain": 0,
            "promoted_to_global": 0,
            "glossary_domain": task_domain,
        }

    summary = request.summary
    if summary is None:
        current = ws.load_terms() if ws.has_terms() else {"summary": "", "terms": []}
        summary = current.get("summary", "")

    to_promote = [
        t
        for t in request.terms
        if t.get("promote_to_domain") or t.get("promote_to_global")
    ]
    promoted = 0
    if to_promote and task_domain:
        csv_before, dict_srcs = _domain_known_src_sets(task_domain)
        promoted = promote_terms_to_domain(to_promote, domain=task_domain)
        promoted_here = {
            str(s).strip().lower()
            for s in (ws.get_kv("task_promoted_srcs", []) or [])
            if str(s).strip()
        }
        for t in to_promote:
            k = str(t.get("src") or "").strip().lower()
            if k and k not in csv_before and k not in dict_srcs:
                promoted_here.add(k)
        ws.set_kv("task_promoted_srcs", sorted(promoted_here))
    saved_count = ws.save_terms(summary, request.terms)
    return {
        "saved": saved_count,
        "promoted_to_domain": promoted,
        "promoted_to_global": promoted,
        "glossary_domain": task_domain,
    }

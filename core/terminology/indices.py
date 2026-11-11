from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any

from core.utils import log_event
from core.workspace import (
    domain_dict_json_path,
    domain_terms_csv_path,
    get_active_workspace,
)

from .custom_terms import (
    TerminologyConfig,
    _file_mtime,
    _file_sig,
    _read_json_cached,
    active_glossary_domain,
    load_custom_terms_from_config,
)
from .lexical import (
    _KB_STOP_TOKENS,
    _is_standard_english,
    _tokenize_surface,
)
from .matcher import _build_term_matcher
from .ngram_index import UnifiedNgramKBIndex
from .normalize import (
    _extract_generic_kb_pairs,
    _normalize_aliases,
    _normalize_kb_entries,
)
from .oov_index import _build_possessive_prefix_map
from .shorthands import _derive_in_video_shorthands

_KB_ENTRIES_LOCK = threading.Lock()
_KB_ENTRIES_CACHE: dict[str, Any] = {"key": None, "data": None}

_KB_INDEX_LOCK = threading.Lock()
_KB_INDEX_CACHE: dict[str, Any] = {"key": None, "data": None}

_TERM_INDEX_LOCK = threading.Lock()
_TERM_INDEX_CACHE: dict[str, Any] = {"key": None, "data": None}


def _kb_cache_key(
    custom_terms_list: list[dict[str, Any]], cfg: TerminologyConfig
) -> tuple[Any, ...]:
    active_domain = active_glossary_domain(cfg)
    dom_dict_path = domain_dict_json_path(active_domain) if active_domain else Path("")
    dom_csv_path = domain_terms_csv_path(active_domain) if active_domain else Path("")
    custom_fp = tuple(
        (
            t.get("src", ""),
            t.get("tgt", ""),
            tuple(t.get("aliases") or []),
            bool(t.get("fix_source", False)),
        )
        for t in custom_terms_list
    )
    return (
        active_domain,
        _file_sig(dom_dict_path) if active_domain else ("", 0, 0),
        _file_sig(dom_csv_path) if active_domain else ("", 0, 0),
        _file_sig(cfg.optional_json_terms_path),
        custom_fp,
    )


def get_kb_entries(
    custom_terms_list: list[dict[str, Any]], cfg: TerminologyConfig
) -> dict[str, dict[str, Any]]:
    cache_key = _kb_cache_key(custom_terms_list, cfg)
    with _KB_ENTRIES_LOCK:
        if (
            _KB_ENTRIES_CACHE["key"] == cache_key
            and _KB_ENTRIES_CACHE["data"] is not None
        ):
            return _KB_ENTRIES_CACHE["data"]

        active_domain = active_glossary_domain(cfg)
        dom_dict_path = (
            domain_dict_json_path(active_domain) if active_domain else Path("")
        )
        kb_entries: dict[str, dict[str, Any]] = {}
        for item in custom_terms_list:
            s = str(item.get("src") or "").strip()
            t = str(item.get("tgt") or "").strip()
            if s and t:
                kb_entries[s.lower()] = {
                    "src": s,
                    "tgt": t,
                    "note": str(item.get("note") or "").strip(),
                    "aliases": _normalize_aliases(item.get("aliases"), s),
                    "fix_source": bool(item.get("fix_source", False)),
                    "source": str(item.get("source") or "custom"),
                }

        if active_domain and dom_dict_path.is_file():
            try:
                raw_dom = _read_json_cached(dom_dict_path)
                _extract_generic_kb_pairs(raw_dom, kb_entries, source_tag="domain_dict")
            except Exception as exc:
                log_event(
                    f"Failed to index domain dict.json ({active_domain}): {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

        if os.path.exists(cfg.optional_json_terms_path):
            try:
                raw_opt = _read_json_cached(cfg.optional_json_terms_path)
                _extract_generic_kb_pairs(raw_opt, kb_entries, source_tag="json_kb")
            except Exception as exc:
                log_event(
                    f"Failed to index {cfg.optional_json_terms_path}: {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

        normalized = _normalize_kb_entries(kb_entries)
        _KB_ENTRIES_CACHE["key"] = cache_key
        _KB_ENTRIES_CACHE["data"] = normalized
        return normalized


def get_kb_index(
    custom_terms_list: list[dict[str, Any]], cfg: TerminologyConfig
) -> dict[str, Any]:
    cache_key = _kb_cache_key(custom_terms_list, cfg)
    with _KB_INDEX_LOCK:
        if _KB_INDEX_CACHE["key"] == cache_key and _KB_INDEX_CACHE["data"] is not None:
            return _KB_INDEX_CACHE["data"]

        kb_entries = get_kb_entries(custom_terms_list, cfg)
        entries_list = list(kb_entries.values())
        matcher = _build_term_matcher(entries_list, include_rare_index=True)
        matcher["possessive_prefix_map"] = _build_possessive_prefix_map(kb_entries)
        ngram_index = UnifiedNgramKBIndex(entries_list)
        data = {
            "entries": kb_entries,
            "matcher": matcher,
            "ngram_index": ngram_index,
        }
        _KB_INDEX_CACHE["key"] = cache_key
        _KB_INDEX_CACHE["data"] = data
        return data


def _merge_term_rows(
    rows: list[dict[str, Any]],
    merged_map: dict[str, dict[str, Any]],
    claimed: set[str] | None = None,
) -> None:
    for item in rows:
        if not isinstance(item, dict):
            continue
        s = str(item.get("src") or "").strip()
        t = str(item.get("tgt") or "").strip()
        if not s or not t:
            continue
        k = s.lower()
        raw_als = _normalize_aliases(item.get("aliases"), s)
        als = [a for a in raw_als if a.lower() not in claimed] if claimed else raw_als
        if k not in merged_map:
            if claimed and k in claimed:
                continue
            merged_map[k] = {
                "src": s,
                "tgt": t,
                "note": str(item.get("note") or "").strip(),
                "aliases": als,
                "fix_source": bool(item.get("fix_source", False)),
                "source": str(item.get("source") or "task"),
            }
        else:
            cur = merged_map[k]
            cur["aliases"] = _normalize_aliases(
                [*(cur.get("aliases") or []), *als], cur["src"]
            )
            cur["fix_source"] = bool(
                cur.get("fix_source", False) or item.get("fix_source", False)
            )


def get_term_index(cfg: TerminologyConfig) -> dict[str, Any]:
    ws = get_active_workspace()
    active_domain = active_glossary_domain(cfg)
    dom_csv = domain_terms_csv_path(active_domain) if active_domain else Path("")
    dom_dict = domain_dict_json_path(active_domain) if active_domain else Path("")
    ws_mtime = ws.get_terms_mtime() if ws.has_terms() else 0.0
    ext_mtime = max(
        _file_mtime(cfg.custom_terms_path),
        _file_mtime(cfg.optional_csv_terms_path),
        _file_mtime(cfg.optional_json_terms_path),
        _file_mtime(cfg.global_glossary_path),
        _file_mtime(dom_csv) if active_domain else 0.0,
        _file_mtime(dom_dict) if active_domain else 0.0,
    )
    utt_rev = (
        ws.get_utterances_revision() if hasattr(ws, "get_utterances_revision") else 0
    )
    cache_key = (
        str(ws.root.resolve()),
        active_domain,
        ws_mtime,
        utt_rev,
        ext_mtime,
        _file_sig(cfg.custom_terms_path),
        _file_sig(cfg.optional_csv_terms_path),
        _file_sig(cfg.optional_json_terms_path),
        _file_sig(cfg.global_glossary_path),
        _file_sig(dom_csv) if active_domain else ("", 0, 0),
        _file_sig(dom_dict) if active_domain else ("", 0, 0),
    )

    with _TERM_INDEX_LOCK:
        if (
            _TERM_INDEX_CACHE["key"] == cache_key
            and _TERM_INDEX_CACHE["data"] is not None
        ):
            return _TERM_INDEX_CACHE["data"]

        task_terms = (ws.load_terms().get("terms") or []) if ws.has_terms() else []
        custom_terms = load_custom_terms_from_config(cfg)
        kb_entries = get_kb_entries(custom_terms, cfg)

        task_is_primary = bool(task_terms) and ws_mtime >= ext_mtime
        primary = task_terms if task_is_primary else custom_terms
        secondary = custom_terms if task_is_primary else task_terms

        merged_map: dict[str, dict[str, Any]] = {}
        _merge_term_rows(primary, merged_map)

        claimed_by_task: set[str] = set()
        if task_is_primary:
            utts = ws.load_utterances() if hasattr(ws, "load_utterances") else []
            utt_text = "\n".join(
                str(u.get("src_punct") or u.get("src") or "").strip()
                for u in utts
                if str(u.get("src_punct") or u.get("src") or "").strip()
            )
            claimed_by_task = _derive_in_video_shorthands(merged_map, utt_text)

        _merge_term_rows(secondary, merged_map, claimed_by_task)

        norm_merged_map = _normalize_kb_entries(merged_map)
        all_entries = list(norm_merged_map.values())
        seen_all = set(norm_merged_map.keys()) | claimed_by_task
        for k, kb_entry in kb_entries.items():
            if k in seen_all:
                continue
            if task_terms and kb_entry.get("source") == "domain_dict":
                bw, _, _ = _tokenize_surface(kb_entry["src"])
                content_bw = [w for w in bw if w not in _KB_STOP_TOKENS]
                if len(content_bw) <= 1 and all(
                    _is_standard_english(w) for w in content_bw
                ):
                    continue
            if claimed_by_task and kb_entry.get("aliases"):
                filtered_als = [
                    a
                    for a in kb_entry["aliases"]
                    if str(a).strip().lower() not in claimed_by_task
                ]
                if len(filtered_als) != len(kb_entry["aliases"]):
                    kb_entry = {**kb_entry, "aliases": filtered_als}
            all_entries.append(kb_entry)

        matcher = _build_term_matcher(all_entries, include_rare_index=False)
        data = {"terms": list(merged_map.values()), "matcher": matcher}
        _TERM_INDEX_CACHE["key"] = cache_key
        _TERM_INDEX_CACHE["data"] = data
        return data

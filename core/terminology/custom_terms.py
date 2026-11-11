from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
from typing import Any

import pandas as pd

from core.utils import log_event
from core.workspace import (
    GLOBAL_GLOSSARY_CSV,
    domain_dict_json_path,
    domain_terms_csv_path,
    load_domain_glossary,
    load_global_glossary,
)

from .normalize import (
    _META_KEYS,
    _coerce_bool,
    _extract_generic_kb_pairs,
    _normalize_aliases,
)


@dataclass(frozen=True, slots=True)
class TerminologyConfig:
    custom_terms_path: str
    optional_csv_terms_path: str
    optional_json_terms_path: str
    global_glossary_path: str
    load_key_or_fn: Callable[..., Any]


def should_load_domain_glossary(cfg: TerminologyConfig) -> bool:
    import core.workspace as _ws

    return (
        cfg.global_glossary_path != str(GLOBAL_GLOSSARY_CSV)
        or cfg.custom_terms_path == "custom_terms.xlsx"
        or _ws.GLOSSARY_DIR != Path("glossary")
    )


def active_glossary_domain(cfg: TerminologyConfig) -> str:
    if not should_load_domain_glossary(cfg):
        return ""
    raw = cfg.load_key_or_fn("glossary_domain", "destiny2")
    if raw is None or not isinstance(raw, str):
        return "destiny2" if raw is None else ""
    return raw.strip().lower()


def _file_mtime(path: str | Path) -> float:
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0
    except OSError:
        return 0.0


def _file_sig(path: str | Path) -> tuple[str, int, int]:
    p = Path(path)
    resolved = str(p.resolve())
    try:
        st = os.stat(p)
        return (resolved, st.st_mtime_ns, st.st_size)
    except OSError:
        return (resolved, 0, 0)


_JSON_FILE_LOCK = threading.Lock()
_JSON_FILE_CACHE: dict[tuple[str, int, int], Any] = {}

_CUSTOM_TERMS_LOCK = threading.Lock()
_CUSTOM_TERMS_CACHE: dict[str, Any] = {"key": None, "data": None}


def _read_json_cached(path: str | Path) -> Any:
    sig = _file_sig(path)
    with _JSON_FILE_LOCK:
        if sig[1] != 0 and sig in _JSON_FILE_CACHE:
            return _JSON_FILE_CACHE[sig]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if sig[1] != 0:
        with _JSON_FILE_LOCK:
            if len(_JSON_FILE_CACHE) >= 16:
                _JSON_FILE_CACHE.clear()
            _JSON_FILE_CACHE[sig] = data
    return data


def _is_missing_cell(val: Any) -> bool:
    if val is None or val is pd.NA:
        return True
    return isinstance(val, float) and math.isnan(val)


def _flat_dict_exceeds_limit(raw_dict: Any, limit: int) -> bool:
    if not isinstance(raw_dict, dict) or len(raw_dict) <= limit:
        return False
    if any(
        k in raw_dict
        for k in ("src", "en", "source", "term", "original", "terms", "entries")
    ):
        return False
    seen: set[str] = set()
    for k, v in raw_dict.items():
        if not isinstance(v, str):
            return False
        s = str(k).strip()
        t = v.replace("\n", "").strip() if "\n" in v else v.strip()
        if (
            s
            and t
            and len(s) <= 80
            and len(t) <= 80
            and s.lower() != t.lower()
            and s.lower() not in _META_KEYS
        ):
            seen.add(s.lower())
            if len(seen) > limit:
                return True
    return False


def _clone_terms_list(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**t, "aliases": list(t.get("aliases") or [])} for t in terms]


def load_custom_terms_from_config(cfg: TerminologyConfig) -> list[dict[str, Any]]:
    active_domain = active_glossary_domain(cfg)
    dom_csv_sig = (
        _file_sig(domain_terms_csv_path(active_domain)) if active_domain else ("", 0, 0)
    )
    dom_dict_sig = (
        _file_sig(domain_dict_json_path(active_domain)) if active_domain else ("", 0, 0)
    )
    cache_key = (
        active_domain,
        dom_csv_sig,
        dom_dict_sig,
        _file_sig(cfg.global_glossary_path),
        _file_sig(cfg.custom_terms_path),
        _file_sig(cfg.optional_csv_terms_path),
        _file_sig(cfg.optional_json_terms_path),
    )
    with _CUSTOM_TERMS_LOCK:
        if (
            _CUSTOM_TERMS_CACHE["key"] == cache_key
            and _CUSTOM_TERMS_CACHE["data"] is not None
        ):
            return _clone_terms_list(_CUSTOM_TERMS_CACHE["data"])

    terms: list[dict[str, Any]] = []
    seen_src: set[str] = set()

    def _add_entry(
        src: Any,
        tgt: Any,
        note: Any = "",
        aliases: Any = None,
        fix_source: Any = False,
        source: str = "custom",
    ) -> None:
        if _is_missing_cell(src) or _is_missing_cell(tgt):
            return
        s = str(src or "").strip()
        t = str(tgt or "").strip()
        if not s or not t or s.lower() == "nan" or t.lower() == "nan":
            return
        sl = s.lower()
        if sl in seen_src:
            return
        seen_src.add(sl)
        note_str = "" if _is_missing_cell(note) else str(note or "").strip()
        if note_str.lower() == "nan":
            note_str = ""
        terms.append(
            {
                "src": s,
                "tgt": t,
                "note": note_str,
                "aliases": []
                if _is_missing_cell(aliases)
                else _normalize_aliases(aliases, s),
                "fix_source": False
                if _is_missing_cell(fix_source)
                else _coerce_bool(fix_source),
                "source": source,
            }
        )

    if active_domain:
        for item in load_domain_glossary(active_domain):
            _add_entry(
                item.get("src", ""),
                item.get("tgt", ""),
                item.get("note", ""),
                item.get("aliases"),
                item.get("fix_source", False),
                source="domain_kb",
            )
        dict_path = domain_dict_json_path(active_domain)
        if dict_path.is_file():
            try:
                raw_dict = _read_json_cached(dict_path)
                if not _flat_dict_exceeds_limit(raw_dict, 500):
                    extracted_dict: dict[str, dict[str, Any]] = {}
                    _extract_generic_kb_pairs(
                        raw_dict,
                        extracted_dict,
                        source_tag="domain_kb",
                        max_entries=500,
                    )
                    if len(extracted_dict) <= 500:
                        for item in extracted_dict.values():
                            _add_entry(
                                item.get("src", ""),
                                item.get("tgt", ""),
                                item.get("note", ""),
                                item.get("aliases"),
                                item.get("fix_source", False),
                                source="domain_kb",
                            )
            except Exception as exc:
                log_event(
                    f"Failed to read domain dict.json ({active_domain}): {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

    global_csv = Path(cfg.global_glossary_path)
    if global_csv.is_file():
        for item in load_global_glossary(global_csv):
            _add_entry(
                item.get("src", ""),
                item.get("tgt", ""),
                item.get("note", ""),
                item.get("aliases"),
                item.get("fix_source", False),
                source="global_kb",
            )

    for tabular_path, reader in (
        (cfg.custom_terms_path, pd.read_excel),
        (cfg.optional_csv_terms_path, lambda p: pd.read_csv(p, encoding="utf-8-sig")),
    ):
        if os.path.exists(tabular_path):
            try:
                df = reader(tabular_path)
                cols = [str(c) for c in df.columns]
                ncols = len(cols)
                col_idx: dict[str, int] = {}
                for idx, cname in enumerate(cols):
                    col_idx.setdefault(cname, idx)
                i_src = col_idx.get("src", 0 if ncols > 0 else -1)
                i_tgt = col_idx.get("tgt", 1 if ncols > 1 else -1)
                i_note = col_idx.get("note", 2 if ncols > 2 else -1)
                i_als = col_idx.get("aliases", 3 if ncols > 3 else -1)
                i_fix = col_idx.get("fix_source", 4 if ncols > 4 else -1)
                for tup in df.itertuples(index=False, name=None):
                    _add_entry(
                        tup[i_src] if i_src >= 0 else "",
                        tup[i_tgt] if i_tgt >= 0 else "",
                        tup[i_note] if i_note >= 0 else "",
                        tup[i_als] if i_als >= 0 else None,
                        tup[i_fix] if i_fix >= 0 else False,
                        source="custom",
                    )
            except Exception as exc:
                log_event(
                    f"Failed to read {tabular_path}: {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

    if os.path.exists(cfg.optional_json_terms_path):
        try:
            data = _read_json_cached(cfg.optional_json_terms_path)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        _add_entry(
                            item.get("src")
                            or item.get("source")
                            or item.get("term", ""),
                            item.get("tgt")
                            or item.get("target")
                            or item.get("translation", ""),
                            item.get("note", ""),
                            item.get("aliases"),
                            item.get("fix_source", False),
                            source="custom",
                        )
            elif isinstance(data, dict):
                raw_terms = data.get("terms")
                if isinstance(raw_terms, list):
                    for item in raw_terms:
                        if isinstance(item, dict):
                            _add_entry(
                                item.get("src", ""),
                                item.get("tgt", ""),
                                item.get("note", ""),
                                item.get("aliases"),
                                item.get("fix_source", False),
                                source="custom",
                            )
                elif all(isinstance(v, str) for v in data.values()):
                    for k, v in data.items():
                        _add_entry(k, v, "", source="custom")
        except Exception as exc:
            log_event(
                f"Failed to read {cfg.optional_json_terms_path}: {exc}",
                level="warning",
                tag="file",
                label="TERMS",
            )

    with _CUSTOM_TERMS_LOCK:
        _CUSTOM_TERMS_CACHE["key"] = cache_key
        _CUSTOM_TERMS_CACHE["data"] = _clone_terms_list(terms)
    return terms

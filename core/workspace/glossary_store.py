from __future__ import annotations

import contextlib
import copy
import csv
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

from core.utils.config_utils import load_key_or
from core.workspace.fs_atomic import (
    register_cleanup_hook,
    replace_with_retry,
    rmtree_robust,
)

DEFAULT_GLOSSARY_DOMAIN = "destiny2"
DOMAIN_TERMS_FILENAME = "terms.csv"
DOMAIN_DICT_FILENAME = "dict.json"

_DOMAIN_NAME_RE = re.compile(r"^[\w\-\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{1,64}$")
_ALIAS_SPLIT_RE = re.compile(r"[,;|/，；｜]")

# Mtime-keyed caches for CSV glossaries and JSON dictionaries.
_CSV_GLOSSARY_CACHE: dict[
    Path, tuple[tuple[int, int, int], list[dict[str, Any]], frozenset[str]]
] = {}
_DICT_META_CACHE: dict[
    Path,
    tuple[
        tuple[int, int, int],
        int,
        list[tuple[str, str, str, str]],
    ],
] = {}
_GLOSSARY_CACHE_LOCK = threading.Lock()


def _purge_glossary_caches(target_resolved: Path) -> None:
    with _GLOSSARY_CACHE_LOCK:
        for p in [
            k
            for k in _CSV_GLOSSARY_CACHE
            if k == target_resolved or target_resolved in k.parents
        ]:
            _CSV_GLOSSARY_CACHE.pop(p, None)
        for p in [
            k
            for k in _DICT_META_CACHE
            if k == target_resolved or target_resolved in k.parents
        ]:
            _DICT_META_CACHE.pop(p, None)


register_cleanup_hook(_purge_glossary_caches)


def _glossary_root(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    import core.workspace as _ws

    return Path(_ws.GLOSSARY_DIR)


def _global_glossary_csv() -> Path:
    import core.workspace as _ws

    return Path(_ws.GLOBAL_GLOSSARY_CSV)


def sanitize_domain_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or not _DOMAIN_NAME_RE.match(cleaned)
    ):
        raise ValueError(f"Invalid glossary domain name: {name!r}")
    return cleaned


def domain_dir_path(domain: str, base_dir: str | Path | None = None) -> Path:
    safe = sanitize_domain_name(domain)
    return _glossary_root(base_dir) / safe


def domain_terms_csv_path(domain: str, base_dir: str | Path | None = None) -> Path:
    return domain_dir_path(domain, base_dir) / DOMAIN_TERMS_FILENAME


def domain_dict_json_path(domain: str, base_dir: str | Path | None = None) -> Path:
    return domain_dir_path(domain, base_dir) / DOMAIN_DICT_FILENAME


def ensure_default_glossary_domain(base_dir: str | Path | None = None) -> Path:
    root = _glossary_root(base_dir)
    d_dir = root / DEFAULT_GLOSSARY_DOMAIN
    if not root.exists():
        d_dir.mkdir(parents=True, exist_ok=True)
        csv_file = d_dir / DOMAIN_TERMS_FILENAME
        save_global_glossary([], csv_path=csv_file)
    return d_dir


def list_glossary_domains(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = _glossary_root(base_dir)
    ensure_default_glossary_domain(root)
    if not root.is_dir():
        return []
    domains: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            sanitize_domain_name(child.name)
        except ValueError:
            continue
        domains.append(get_glossary_domain_meta(child.name, base_dir=root))
    return domains


def _load_cached_dict_meta(
    dict_path: Path, *, strict: bool = False
) -> tuple[int, list[tuple[str, str, str, str]]]:
    try:
        st = dict_path.stat()
    except OSError:
        if strict:
            raise
        return 0, []
    resolved = dict_path.resolve()
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _GLOSSARY_CACHE_LOCK:
        cached = _DICT_META_CACHE.get(resolved)
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]

    try:
        raw_data = json.loads(dict_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        if strict:
            raise
        return 0, []

    raw_count = 0
    pairs: list[tuple[str, str, str, str]] = []
    if isinstance(raw_data, dict):
        inner_entries = raw_data.get("entries")
        inner_terms = raw_data.get("terms")
        if isinstance(inner_entries, dict):
            raw_count = len(inner_entries)
            src_iter = inner_entries.items()
        elif isinstance(inner_terms, list):
            raw_count = len(inner_terms)
            src_iter = None
            for item in inner_terms:
                if isinstance(item, dict):
                    s = str(item.get("src") or item.get("en") or "").strip()
                    t = str(item.get("tgt") or item.get("zh") or "").strip()
                    if s and t:
                        pairs.append((s, t, s.lower(), t.lower()))
        else:
            raw_count = len(raw_data)
            src_iter = raw_data.items()
        if src_iter is not None:
            for k, v in src_iter:
                s = str(k).strip()
                if isinstance(v, str):
                    t = v.strip()
                elif isinstance(v, dict):
                    t = str(v.get("tgt") or v.get("zh") or "").strip()
                else:
                    t = ""
                if s and t:
                    pairs.append((s, t, s.lower(), t.lower()))
    elif isinstance(raw_data, list):
        raw_count = len(raw_data)
        for item in raw_data:
            if isinstance(item, dict):
                s = str(item.get("src") or item.get("en") or "").strip()
                t = str(item.get("tgt") or item.get("zh") or "").strip()
                if s and t:
                    pairs.append((s, t, s.lower(), t.lower()))

    with _GLOSSARY_CACHE_LOCK:
        _DICT_META_CACHE[resolved] = (sig, raw_count, pairs)
    return raw_count, pairs


def count_domain_dict_entries(dict_path: str | Path) -> int:
    p = Path(dict_path)
    if not p.is_file():
        return 0
    raw_count, _ = _load_cached_dict_meta(p)
    return raw_count


def search_domain_dict_entries(
    domain: str,
    q: str = "",
    limit: int = 50,
    base_dir: str | Path | None = None,
) -> tuple[int, list[dict[str, str]] | None]:
    """Search mtime-cached `dict.json` entries without re-reading or re-parsing JSON per keystroke."""
    dict_file = domain_dict_json_path(domain, base_dir=base_dir)
    if not dict_file.is_file():
        return 0, None
    _, pairs = _load_cached_dict_meta(dict_file, strict=True)
    bounded = max(1, min(limit, 200))
    q_norm = str(q or "").strip().lower()
    if not q_norm:
        return len(pairs), [{"k": s, "v": t} for s, t, _, _ in pairs[:bounded]]
    matched_count = 0
    out: list[dict[str, str]] = []
    for s, t, s_low, t_low in pairs:
        if q_norm in s_low or q_norm in t_low:
            matched_count += 1
            if len(out) < bounded:
                out.append({"k": s, "v": t})
    return matched_count, out


def get_glossary_domain_meta(
    domain: str, base_dir: str | Path | None = None
) -> dict[str, Any]:
    safe = sanitize_domain_name(domain)
    d_dir = domain_dir_path(safe, base_dir)
    csv_file = d_dir / DOMAIN_TERMS_FILENAME
    dict_file = d_dir / DOMAIN_DICT_FILENAME
    terms_count = len(_load_cached_csv_glossary(csv_file)) if csv_file.is_file() else 0
    has_dict = dict_file.is_file()
    dict_count = count_domain_dict_entries(dict_file) if has_dict else 0
    return {
        "domain": safe,
        "name": safe,
        "terms_count": terms_count,
        "has_dict": has_dict,
        "dict_count": dict_count,
    }


def create_glossary_domain(
    domain: str, base_dir: str | Path | None = None
) -> dict[str, Any]:
    safe = sanitize_domain_name(domain)
    d_dir = domain_dir_path(safe, base_dir)
    d_dir.mkdir(parents=True, exist_ok=True)
    csv_file = d_dir / DOMAIN_TERMS_FILENAME
    if not csv_file.is_file():
        save_global_glossary([], csv_path=csv_file)
    return get_glossary_domain_meta(safe, base_dir=base_dir)


def delete_glossary_domain(domain: str, base_dir: str | Path | None = None) -> bool:
    safe = sanitize_domain_name(domain)
    d_dir = domain_dir_path(safe, base_dir)
    if not d_dir.is_dir():
        return False
    rmtree_robust(d_dir)
    return True


def load_domain_glossary(
    domain: str, base_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    if not domain or not str(domain).strip():
        return []
    csv_file = domain_terms_csv_path(domain, base_dir=base_dir)
    return load_global_glossary(csv_file)


def domain_csv_known_srcs(
    domain: str, base_dir: str | Path | None = None
) -> frozenset[str]:
    if not domain or not str(domain).strip():
        return frozenset()
    csv_file = domain_terms_csv_path(domain, base_dir=base_dir)
    if not csv_file.is_file():
        return frozenset()
    _, srcs = _load_cached_csv_glossary_with_srcs(csv_file)
    return srcs


def save_domain_glossary(
    domain: str,
    entries: list[dict[str, Any]],
    base_dir: str | Path | None = None,
) -> None:
    csv_file = domain_terms_csv_path(domain, base_dir=base_dir)
    save_global_glossary(entries, csv_path=csv_file)


def promote_terms_to_domain(
    terms: list[dict[str, Any]],
    domain: str | None = None,
    base_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> int:
    if csv_path is not None:
        return promote_terms_to_global(terms, csv_path=csv_path)
    target_domain = str(
        domain
        if domain is not None
        else load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN)
    ).strip()
    if not target_domain:
        return 0
    csv_file = domain_terms_csv_path(target_domain, base_dir=base_dir)
    return promote_terms_to_global(terms, csv_path=csv_file)


def remove_terms_from_domain(
    srcs: list[str],
    domain: str | None = None,
    base_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> int:
    if csv_path is not None:
        return remove_terms_from_global(srcs, csv_path=csv_path)
    target_domain = str(
        domain
        if domain is not None
        else load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN)
    ).strip()
    if not target_domain:
        return 0
    csv_file = domain_terms_csv_path(target_domain, base_dir=base_dir)
    return remove_terms_from_global(srcs, csv_path=csv_file)


def _resolve_default_glossary_csv() -> Path:
    global_csv = _global_glossary_csv()
    if global_csv != Path("glossary") / "custom_terms.csv" or global_csv.is_file():
        return global_csv
    active_domain = str(
        load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
    ).strip()
    if active_domain:
        with contextlib.suppress(ValueError):
            return domain_terms_csv_path(active_domain)
    return global_csv


def _load_cached_csv_glossary_with_srcs(
    p: Path,
) -> tuple[list[dict[str, Any]], frozenset[str]]:
    try:
        st = p.stat()
    except OSError:
        return [], frozenset()
    resolved = p.resolve()
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _GLOSSARY_CACHE_LOCK:
        cached = _CSV_GLOSSARY_CACHE.get(resolved)
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]

    entries: list[dict[str, Any]] = []
    srcs: set[str] = set()
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = str(row.get("src") or row.get("Original") or "").strip()
            tgt = str(row.get("tgt") or row.get("Translation") or "").strip()
            if not src or not tgt:
                continue
            raw_aliases = str(row.get("aliases") or "").strip()
            aliases = (
                [a.strip() for a in _ALIAS_SPLIT_RE.split(raw_aliases) if a.strip()]
                if raw_aliases
                else []
            )
            srcs.add(src.lower())
            entries.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "aliases": aliases,
                    "note": str(row.get("note") or row.get("Note") or "").strip(),
                    "fix_source": str(row.get("fix_source") or "").strip().lower()
                    in {"1", "true", "yes"},
                    "source": "global_kb",
                }
            )
    frozen_srcs = frozenset(srcs)
    with _GLOSSARY_CACHE_LOCK:
        _CSV_GLOSSARY_CACHE[resolved] = (sig, entries, frozen_srcs)
    return entries, frozen_srcs


def _load_cached_csv_glossary(p: Path) -> list[dict[str, Any]]:
    entries, _ = _load_cached_csv_glossary_with_srcs(p)
    return entries


def load_global_glossary(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    global_csv = _global_glossary_csv()
    if csv_path is None:
        if (
            global_csv == Path("glossary") / "custom_terms.csv"
            and not global_csv.is_file()
        ):
            active_domain = str(
                load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
            ).strip()
            if not active_domain:
                return []
            p = _resolve_default_glossary_csv()
        else:
            p = global_csv
    else:
        p = Path(csv_path)
    if not p.is_file():
        return []
    return copy.deepcopy(_load_cached_csv_glossary(p))


def save_global_glossary(
    entries: list[dict[str, Any]], csv_path: str | Path | None = None
) -> None:
    p = Path(csv_path) if csv_path is not None else _resolve_default_glossary_csv()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_name(f".{p.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    fieldnames = ["src", "tgt", "aliases", "note", "fix_source"]
    normalized_saved: list[dict[str, Any]] = []
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in entries:
                src = str(item.get("src") or "").strip()
                tgt = str(item.get("tgt") or "").strip()
                if not src or not tgt:
                    continue
                aliases = item.get("aliases") or []
                if isinstance(aliases, list):
                    clean_aliases = [str(a).strip() for a in aliases if str(a).strip()]
                    aliases_str = "|".join(clean_aliases)
                else:
                    aliases_str = str(aliases).strip()
                    clean_aliases = [
                        a.strip()
                        for a in _ALIAS_SPLIT_RE.split(aliases_str)
                        if a.strip()
                    ]
                note = str(item.get("note") or "").strip()
                fix_src = bool(item.get("fix_source"))
                writer.writerow(
                    {
                        "src": src,
                        "tgt": tgt,
                        "aliases": aliases_str,
                        "note": note,
                        "fix_source": "true" if fix_src else "false",
                    }
                )
                normalized_saved.append(
                    {
                        "src": src,
                        "tgt": tgt,
                        "aliases": clean_aliases,
                        "note": note,
                        "fix_source": fix_src,
                        "source": "global_kb",
                    }
                )
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(tmp_path, p)
        with contextlib.suppress(OSError):
            st = p.stat()
            frozen_srcs = frozenset(item["src"].lower() for item in normalized_saved)
            with _GLOSSARY_CACHE_LOCK:
                _CSV_GLOSSARY_CACHE[p.resolve()] = (
                    (st.st_ino, st.st_mtime_ns, st.st_size),
                    normalized_saved,
                    frozen_srcs,
                )
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def promote_terms_to_global(
    terms: list[dict[str, Any]], csv_path: str | Path | None = None
) -> int:
    global_csv = _global_glossary_csv()
    if (
        csv_path is None
        and global_csv == Path("glossary") / "custom_terms.csv"
        and not global_csv.is_file()
    ):
        active_domain = str(
            load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
        ).strip()
        if not active_domain:
            return 0
    existing = load_global_glossary(csv_path)
    by_key = {item["src"].lower(): item for item in existing}
    added_or_updated = 0
    for t in terms:
        src = str(t.get("src") or "").strip()
        tgt = str(t.get("tgt") or "").strip()
        if not src or not tgt:
            continue
        key = src.lower()
        prev_aliases = by_key.get(key, {}).get("aliases") or []
        curr_aliases = t.get("aliases")
        curr_list = curr_aliases if isinstance(curr_aliases, list) else []
        merged_aliases = list(dict.fromkeys([*prev_aliases, *curr_list]))
        by_key[key] = {
            "src": src,
            "tgt": tgt,
            "aliases": merged_aliases,
            "note": str(t.get("note") or by_key.get(key, {}).get("note") or "").strip(),
            "fix_source": bool(
                t.get("fix_source", by_key.get(key, {}).get("fix_source", False))
            ),
            "source": "global_kb",
        }
        added_or_updated += 1
    save_global_glossary(list(by_key.values()), csv_path)
    return added_or_updated


def remove_terms_from_global(
    srcs: list[str], csv_path: str | Path | None = None
) -> int:
    target_keys = {str(s or "").strip().lower() for s in srcs if str(s or "").strip()}
    if not target_keys:
        return 0
    existing = load_global_glossary(csv_path)
    if not existing:
        return 0
    kept = [item for item in existing if item["src"].lower() not in target_keys]
    removed = len(existing) - len(kept)
    if removed > 0:
        save_global_glossary(kept, csv_path)
    return removed

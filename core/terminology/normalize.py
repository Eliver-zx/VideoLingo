from __future__ import annotations

import re
from typing import Any
import unicodedata

from .lexical import (
    _KB_STOP_TOKENS,
    _is_standard_english,
    _tokenize_surface,
)

_ALIAS_SPLIT_RE = re.compile(r"[|,;\n]+")
_NEWLINE_STRIP_RE = re.compile(r"\s*\n\s*")
_COLON_SPLIT_RE = re.compile(r"[：:]")
_META_KEYS = frozenset(("version", "language", "type", "id", "name"))


def _is_inflection_of_src(alias_str: str, src_str: str) -> bool:
    if not src_str or not alias_str:
        return False
    src_ascii = (
        unicodedata.normalize("NFKD", src_str)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    al_ascii = (
        unicodedata.normalize("NFKD", alias_str)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    if src_ascii != src_str.lower() and al_ascii == alias_str.lower():
        return False
    _, src_stems, _ = _tokenize_surface(src_str)
    _, al_stems, _ = _tokenize_surface(alias_str)
    return bool(src_stems and src_stems == al_stems)


def _is_superphrase_of_src(alias_str: str, src_str: str) -> bool:
    _, src_stems, _ = _tokenize_surface(src_str)
    _, al_stems, _ = _tokenize_surface(alias_str)
    n_src, n_al = len(src_stems), len(al_stems)
    if not n_src or n_al <= n_src:
        return False
    return any(al_stems[i : i + n_src] == src_stems for i in range(n_al - n_src + 1))


def _normalize_aliases(raw_aliases: Any, src_str: str = "") -> list[str]:
    if not raw_aliases:
        return []
    if isinstance(raw_aliases, str):
        parts = [p.strip() for p in _ALIAS_SPLIT_RE.split(raw_aliases) if p.strip()]
    elif isinstance(raw_aliases, (list, tuple, set)):
        parts = [
            str(p).strip()
            for p in raw_aliases
            if p is not None and str(p).strip() and str(p).strip().lower() != "nan"
        ]
    else:
        return []
    src_low = src_str.strip().lower()
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p.lower() == src_low or p in seen:
            continue
        if src_str and (
            _is_inflection_of_src(p, src_str) or _is_superphrase_of_src(p, src_str)
        ):
            continue
        seen.add(p)
        out.append(p)
    return out


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _clean_tgt_text(val: str) -> str:
    return (_NEWLINE_STRIP_RE.sub("", val) if "\n" in val else val).strip()


def _extract_generic_kb_pairs(
    node: Any,
    out: dict[str, dict[str, Any]],
    tgt_lang_hint: str = "zh",
    source_tag: str = "kb",
    max_entries: int | None = None,
) -> None:
    if max_entries is not None and len(out) > max_entries:
        return
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
            t = _clean_tgt_text(tgt_val)
            if s and t and s.lower() != t.lower():
                key = s.lower()
                if key not in out:
                    out[key] = {
                        "src": s,
                        "tgt": t,
                        "note": str(node.get("note") or "").strip(),
                        "aliases": _normalize_aliases(node.get("aliases"), s),
                        "fix_source": _coerce_bool(node.get("fix_source", False)),
                        "source": str(node.get("source") or source_tag),
                    }
                return
        for k, v in node.items():
            if max_entries is not None and len(out) > max_entries:
                return
            if isinstance(v, str):
                s = str(k).strip()
                t = _clean_tgt_text(v)
                if (
                    s
                    and t
                    and len(s) <= 80
                    and len(t) <= 80
                    and s.lower() != t.lower()
                    and s.lower() not in _META_KEYS
                ):
                    key = s.lower()
                    if key not in out:
                        out[key] = {
                            "src": s,
                            "tgt": t,
                            "note": "",
                            "aliases": [],
                            "fix_source": False,
                            "source": source_tag,
                        }
            elif isinstance(v, (dict, list)):
                _extract_generic_kb_pairs(
                    v,
                    out,
                    tgt_lang_hint,
                    source_tag=source_tag,
                    max_entries=max_entries,
                )
    elif isinstance(node, list):
        for item in node:
            if max_entries is not None and len(out) > max_entries:
                return
            if isinstance(item, (dict, list)):
                _extract_generic_kb_pairs(
                    item,
                    out,
                    tgt_lang_hint,
                    source_tag=source_tag,
                    max_entries=max_entries,
                )


def _normalize_kb_entries(
    kb_entries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, entry in kb_entries.items():
        clean_tgt = _clean_tgt_text(str(entry.get("tgt") or ""))
        als = list(entry.get("aliases") or [])
        src = str(entry.get("src") or "").strip()
        if "-" in src:
            als_low = {a.lower() for a in als}
            for part in src.split():
                if (
                    "-" in part[1:-1]
                    and part[0].isalnum()
                    and len(part) >= 6
                    and part.lower() != src.lower()
                    and part.lower() not in als_low
                ):
                    als.append(part)
                    als_low.add(part.lower())
        if src.lower().startswith("the "):
            no_the = src[4:].strip()
            words = src.split()
            if no_the and (
                len(words) >= 3
                or (
                    len(words) == 2
                    and len(no_the) >= 6
                    and not _is_standard_english(no_the.lower())
                )
            ):
                if no_the.lower() not in {a.lower() for a in als}:
                    als.append(no_the)
        out[k] = {
            **entry,
            "src": src,
            "tgt": clean_tgt,
            "aliases": _normalize_aliases(als, src) if als else [],
        }

    for k, entry in list(out.items()):
        src = entry["src"]
        tgt = entry["tgt"]
        if ":" in src and ("：" in tgt or ":" in tgt):
            s_parts = [p.strip() for p in src.split(":") if p.strip()]
            t_parts = [p.strip() for p in _COLON_SPLIT_RE.split(tgt) if p.strip()]
            if len(s_parts) == 2 and len(t_parts) == 2:
                _, tail_s = s_parts
                pfx_t, tail_t = t_parts
                bw, _, _ = _tokenize_surface(tail_s)
                content_bw = [w for w in bw if w not in _KB_STOP_TOKENS]
                if len(content_bw) >= 2 or (
                    len(content_bw) == 1 and not _is_standard_english(content_bw[0])
                ):
                    tk = tail_s.lower()
                    if tk not in out:
                        out[tk] = {
                            "src": tail_s,
                            "tgt": tail_t,
                            "note": pfx_t,
                            "aliases": [],
                            "fix_source": False,
                            "source": entry.get("source", "domain_dict"),
                        }
    return out

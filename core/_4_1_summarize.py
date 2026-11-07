from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
import unicodedata

from nltk.stem import SnowballStemmer
import pandas as pd

from core.concurrency import ContextThreadPoolExecutor
from core.prompts import get_summary_prompt
from core.utils import (
    ask_gpt,
    check_cancel,
    check_file_exists,
    load_key_or,
    log_event,
    report_progress,
)
from core.workspace import (
    GLOBAL_GLOSSARY_CSV,
    domain_dict_json_path,
    domain_terms_csv_path,
    get_active_workspace,
    load_domain_glossary,
    load_global_glossary,
)

CUSTOM_TERMS_PATH = "custom_terms.xlsx"
OPTIONAL_CSV_TERMS_PATH = "custom_terms.csv"
OPTIONAL_JSON_TERMS_PATH = "term-map.json"
GLOBAL_GLOSSARY_PATH = str(GLOBAL_GLOSSARY_CSV)
SUMMARY_CHUNK_WEIGHTED_CHARS = 8000

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
_SIGNIFICANT_SYMBOL_RE = re.compile(r"(^\.[A-Za-z0-9]+|[+#@$%&])")
_STEMMER = SnowballStemmer("english")

_KB_STOP_TOKENS: frozenset[str] = frozenset(
    (
        "a an and all are as at away be but by can do for from get go had has "
        "have he her his how if in into is it its just ll me my no not now of "
        "on one only or our out over re so some such than that the their them "
        "then there these they this to too up us ve very was we were what when "
        "where which while who will with would you your"
    ).split()
)


@lru_cache(maxsize=65536)
def _stem_word(word: str) -> str:
    w = word.lower()
    if len(w) <= 2 or w.endswith("ism"):
        return w
    st = _STEMMER.stem(w)
    if st.endswith("lv") and len(st) >= 4:
        st = st[:-1] + "f"
    elif len(st) >= 3 and st[-1] == st[-2] and st[-1] in "flsz":
        st = st[:-1]
    return st


def _tokenize_surface(text: str) -> tuple[list[str], list[str], list[str]]:
    folded = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    words = _WORD_RE.findall(folded)
    base_words: list[str] = []
    raw_words: list[str] = []
    for w in words:
        bw = re.sub(r"['’]s$", "", w).replace("'", "").replace("’", "")
        rw = w.replace("'", "").replace("’", "")
        if bw:
            base_words.append(bw)
            raw_words.append(rw)
    stems = [_stem_word(w) for w in base_words]
    return base_words, stems, raw_words


def _tokenize_and_normalize(text: str) -> tuple[list[str], list[str]]:
    base_words, stems, _ = _tokenize_surface(text)
    return base_words, stems


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
        parts = [p.strip() for p in re.split(r"[|,;\n]+", raw_aliases) if p.strip()]
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


def _extract_generic_kb_pairs(
    node: Any,
    out: dict[str, dict[str, Any]],
    tgt_lang_hint: str = "zh",
    source_tag: str = "kb",
) -> None:
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
            t = tgt_val.strip()
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
            if isinstance(v, str):
                s = str(k).strip()
                t = v.strip()
                if (
                    s
                    and t
                    and len(s) <= 80
                    and len(t) <= 80
                    and s.lower() != t.lower()
                    and s.lower() not in ("version", "language", "type", "id", "name")
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
                _extract_generic_kb_pairs(v, out, tgt_lang_hint, source_tag=source_tag)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _extract_generic_kb_pairs(
                    item, out, tgt_lang_hint, source_tag=source_tag
                )


def _should_load_domain_glossary() -> bool:
    import core.workspace as _ws

    return (
        GLOBAL_GLOSSARY_PATH != str(GLOBAL_GLOSSARY_CSV)
        or CUSTOM_TERMS_PATH == "custom_terms.xlsx"
        or _ws.GLOSSARY_DIR != Path("glossary")
    )


def _active_glossary_domain() -> str:
    if not _should_load_domain_glossary():
        return ""
    raw = load_key_or("glossary_domain", "destiny2")
    if raw is None or not isinstance(raw, str):
        return "destiny2" if raw is None else ""
    return raw.strip().lower()


def load_custom_terms() -> list[dict[str, Any]]:
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
        s = str(src or "").strip()
        t = str(tgt or "").strip()
        if not s or not t or s.lower() == "nan" or t.lower() == "nan":
            return
        sl = s.lower()
        if sl in seen_src:
            return
        seen_src.add(sl)
        terms.append(
            {
                "src": s,
                "tgt": t,
                "note": ""
                if str(note or "").strip().lower() == "nan"
                else str(note or "").strip(),
                "aliases": _normalize_aliases(aliases, s),
                "fix_source": _coerce_bool(fix_source),
                "source": source,
            }
        )

    active_domain = _active_glossary_domain()
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
                raw_dict = json.loads(dict_path.read_text(encoding="utf-8"))
                extracted_dict: dict[str, dict[str, Any]] = {}
                _extract_generic_kb_pairs(
                    raw_dict, extracted_dict, source_tag="domain_kb"
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

    global_csv = Path(GLOBAL_GLOSSARY_PATH)
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
        (CUSTOM_TERMS_PATH, pd.read_excel),
        (OPTIONAL_CSV_TERMS_PATH, lambda p: pd.read_csv(p, encoding="utf-8-sig")),
    ):
        if os.path.exists(tabular_path):
            try:
                df = reader(tabular_path)
                for _, row in df.iterrows():
                    _add_entry(
                        row.get("src", row.iloc[0] if len(row) > 0 else ""),
                        row.get("tgt", row.iloc[1] if len(row) > 1 else ""),
                        row.get("note", row.iloc[2] if len(row) > 2 else ""),
                        row.get("aliases", row.iloc[3] if len(row) > 3 else None),
                        row.get("fix_source", row.iloc[4] if len(row) > 4 else False),
                        source="custom",
                    )
            except Exception as exc:
                log_event(
                    f"Failed to read {tabular_path}: {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

    if os.path.exists(OPTIONAL_JSON_TERMS_PATH):
        try:
            with open(OPTIONAL_JSON_TERMS_PATH, encoding="utf-8") as f:
                data = json.load(f)
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
                f"Failed to read {OPTIONAL_JSON_TERMS_PATH}: {exc}",
                level="warning",
                tag="file",
                label="TERMS",
            )

    return terms


@dataclass(frozen=True, slots=True)
class SurfaceRule:
    entry: dict[str, Any]
    surface: str
    is_canonical_src: bool
    sig: tuple[str, str]


@dataclass(slots=True)
class TermHit:
    entry: dict[str, Any]
    src_matched: bool = False
    matched_aliases: list[str] = field(default_factory=list)
    first_token_idx: int = 0


def _append_rule(
    bucket_map: dict[Any, list[SurfaceRule]],
    key: Any,
    rule: SurfaceRule,
) -> None:
    bucket = bucket_map.get(key)
    if bucket is None:
        bucket_map[key] = [rule]
        return
    sig = rule.sig
    for r in bucket:
        if r.sig == sig:
            return
    bucket.append(rule)


def _build_term_matcher(
    entries: list[dict[str, Any]], max_rare_token_df: int = 8
) -> dict[str, Any]:
    exact_ngram_map: dict[tuple[str, ...], list[SurfaceRule]] = {}
    stem_ngram_map: dict[tuple[str, ...], list[SurfaceRule]] = {}
    first_stem_lengths: dict[str, set[int]] = {}
    compact_stem_map: dict[str, list[SurfaceRule]] = {}
    symbol_cjk_terms: list[tuple[str, re.Pattern[str] | None, SurfaceRule]] = []
    raw_token_inv: dict[str, list[dict[str, Any]]] = {}

    for entry in entries:
        src = str(entry.get("src") or "").strip()
        tgt = str(entry.get("tgt") or "").strip()
        if not src or not tgt:
            continue
        sig = (src.lower(), tgt.lower())
        aliases = _normalize_aliases(entry.get("aliases"), src)
        surface_specs = [(src, True), *((al, False) for al in aliases)]

        for surf_str, is_src in surface_specs:
            s_clean = surf_str.strip()
            if not s_clean:
                continue
            rule = SurfaceRule(
                entry=entry, surface=s_clean, is_canonical_src=is_src, sig=sig
            )
            if _CJK_RE.search(s_clean) or _SIGNIFICANT_SYMBOL_RE.search(s_clean):
                pat = None
                if not _CJK_RE.search(s_clean):
                    pat = re.compile(
                        rf"(?<![A-Za-z0-9]){re.escape(s_clean.lower())}(?![A-Za-z0-9])"
                    )
                symbol_cjk_terms.append((s_clean.lower(), pat, rule))
                continue

            bw, st, rw = _tokenize_surface(s_clean)
            if not st or len(st) > 8:
                continue
            if all(w in _KB_STOP_TOKENS for w in bw):
                continue

            _append_rule(exact_ngram_map, tuple(bw), rule)
            _append_rule(stem_ngram_map, tuple(st), rule)
            first_stem_lengths.setdefault(st[0], set()).add(len(st))

            joined_bw = "".join(bw)
            joined_rw = "".join(rw)
            joined_st = "".join(bw[:-1] + [st[-1]])
            for ck in (
                joined_bw,
                joined_rw,
                joined_st,
                _stem_word(joined_bw),
                _stem_word(joined_rw),
            ):
                if len(ck) >= 3 and ck not in _KB_STOP_TOKENS:
                    _append_rule(compact_stem_map, ck, rule)

            if is_src:
                for tok in set(st):
                    if len(tok) >= 4 and tok not in _KB_STOP_TOKENS:
                        t_bucket = raw_token_inv.get(tok)
                        if t_bucket is None:
                            raw_token_inv[tok] = [entry]
                        elif len(t_bucket) <= max_rare_token_df:
                            if not any(x["src"] == src for x in t_bucket):
                                t_bucket.append(entry)

    rare_token_inv = {
        tok: tuple(bucket)
        for tok, bucket in raw_token_inv.items()
        if 1 <= len(bucket) <= max_rare_token_df
    }

    return {
        "exact_ngram_map": {k: tuple(v) for k, v in exact_ngram_map.items()},
        "stem_ngram_map": {k: tuple(v) for k, v in stem_ngram_map.items()},
        "first_stem_lengths": {k: frozenset(v) for k, v in first_stem_lengths.items()},
        "compact_stem_map": {k: tuple(v) for k, v in compact_stem_map.items()},
        "symbol_cjk_terms": tuple(symbol_cjk_terms),
        "rare_token_inv": rare_token_inv,
    }


def _match_terms_in_text(text: str, matcher: dict[str, Any]) -> list[TermHit]:
    if not text or not text.strip():
        return []

    exact_ngram_map: dict[tuple[str, ...], tuple[SurfaceRule, ...]] = matcher.get(
        "exact_ngram_map", {}
    )
    stem_ngram_map: dict[tuple[str, ...], tuple[SurfaceRule, ...]] = matcher[
        "stem_ngram_map"
    ]
    first_stem_lengths: dict[str, frozenset[int]] = matcher["first_stem_lengths"]
    compact_stem_map: dict[str, tuple[SurfaceRule, ...]] = matcher["compact_stem_map"]
    symbol_cjk_terms = matcher["symbol_cjk_terms"]

    hits_by_sig: dict[tuple[str, str], TermHit] = {}

    def _record_channel(
        rule: SurfaceRule,
        token_idx: int,
        via_compact_channel: bool = False,
        span_text: str = "",
    ) -> None:
        sig = rule.sig
        hit = hits_by_sig.get(sig)
        if hit is None:
            hit = TermHit(
                entry=rule.entry,
                src_matched=False,
                matched_aliases=[],
                first_token_idx=token_idx,
            )
            hits_by_sig[sig] = hit
        elif token_idx < hit.first_token_idx:
            hit.first_token_idx = token_idx

        if not via_compact_channel and rule.is_canonical_src:
            hit.src_matched = True
        else:
            alias_str = (
                span_text.strip()
                if (via_compact_channel and rule.is_canonical_src)
                else rule.surface.strip()
            )
            if (
                alias_str
                and alias_str.lower() != rule.entry["src"].lower()
                and alias_str.lower() not in {a.lower() for a in hit.matched_aliases}
            ):
                hit.matched_aliases.append(alias_str)

    text_lower = text.lower()
    for s_low, pat, rule in symbol_cjk_terms:
        if s_low in text_lower:
            if pat is None or pat.search(text_lower):
                _record_channel(rule, token_idx=0, via_compact_channel=False)

    folded = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    orig_words = _WORD_RE.findall(folded)
    if not orig_words:
        return sorted(hits_by_sig.values(), key=lambda h: h.first_token_idx)

    base_words: list[str] = []
    raw_words: list[str] = []
    for w in orig_words:
        wl = w.lower()
        bw = re.sub(r"['’]s$", "", wl).replace("'", "").replace("’", "")
        rw = wl.replace("'", "").replace("’", "")
        base_words.append(bw)
        raw_words.append(rw)
    stems = [_stem_word(bw) if bw else "" for bw in base_words]
    valid_comp = [len(bw) >= 3 and bw not in _KB_STOP_TOKENS for bw in base_words]

    n = len(stems)
    covered = [False] * n

    for k in range(8, 0, -1):
        for i in range(n - k + 1):
            if covered[i]:
                continue
            lens = first_stem_lengths.get(stems[i])
            if lens and k in lens and not any(covered[i : i + k]):
                bucket = exact_ngram_map.get(
                    tuple(base_words[i : i + k])
                ) or stem_ngram_map.get(tuple(stems[i : i + k]))
                if bucket:
                    for j in range(i, i + k):
                        covered[j] = True
                    span = " ".join(orig_words[i : i + k])
                    for rule in bucket:
                        _record_channel(
                            rule, i, via_compact_channel=False, span_text=span
                        )
                    continue

            if k == 1 and not covered[i]:
                c_bucket = (
                    compact_stem_map.get(base_words[i])
                    or compact_stem_map.get(stems[i])
                    or compact_stem_map.get(raw_words[i])
                )
                if c_bucket:
                    covered[i] = True
                    for rule in c_bucket:
                        _record_channel(
                            rule,
                            i,
                            via_compact_channel=True,
                            span_text=orig_words[i],
                        )
            elif (
                2 <= k <= 3
                and valid_comp[i]
                and valid_comp[i + 1]
                and (k == 2 or valid_comp[i + 2])
                and not any(covered[i : i + k])
            ):
                span_bw = base_words[i : i + k]
                c_bucket = compact_stem_map.get(
                    "".join(span_bw[:-1]) + stems[i + k - 1]
                ) or compact_stem_map.get("".join(span_bw))
                if c_bucket:
                    for j in range(i, i + k):
                        covered[j] = True
                    span = " ".join(orig_words[i : i + k])
                    for rule in c_bucket:
                        _record_channel(
                            rule, i, via_compact_channel=True, span_text=span
                        )

    return sorted(hits_by_sig.values(), key=lambda h: h.first_token_idx)


def format_term_hits_for_prompt(hits: list[TermHit]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        src = hit.entry["src"]
        tgt = hit.entry["tgt"]
        for line in (
            *([f"{src}:{tgt}"] if hit.src_matched else []),
            *(f"{al}:{tgt} ({src})" for al in hit.matched_aliases),
        ):
            k = line.lower()
            if k not in seen:
                seen.add(k)
                lines.append(line)
    return lines


_KB_INDEX_LOCK = threading.Lock()
_KB_INDEX_CACHE: dict[str, Any] = {"key": None, "data": None}


def _file_mtime(path: str | Path) -> float:
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0
    except OSError:
        return 0.0


def _get_kb_index(custom_terms_list: list[dict[str, Any]]) -> dict[str, Any]:
    active_domain = _active_glossary_domain()
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
    cache_key = (
        active_domain,
        _file_mtime(dom_dict_path) if active_domain else 0.0,
        _file_mtime(dom_csv_path) if active_domain else 0.0,
        _file_mtime(OPTIONAL_JSON_TERMS_PATH),
        custom_fp,
    )
    with _KB_INDEX_LOCK:
        if _KB_INDEX_CACHE["key"] == cache_key and _KB_INDEX_CACHE["data"] is not None:
            return _KB_INDEX_CACHE["data"]

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
                raw_dom = json.loads(dom_dict_path.read_text(encoding="utf-8"))
                _extract_generic_kb_pairs(raw_dom, kb_entries, source_tag="domain_dict")
            except Exception as exc:
                log_event(
                    f"Failed to index domain dict.json ({active_domain}): {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

        if os.path.exists(OPTIONAL_JSON_TERMS_PATH):
            try:
                with open(OPTIONAL_JSON_TERMS_PATH, encoding="utf-8") as f:
                    raw_opt = json.load(f)
                _extract_generic_kb_pairs(raw_opt, kb_entries, source_tag="json_kb")
            except Exception as exc:
                log_event(
                    f"Failed to index {OPTIONAL_JSON_TERMS_PATH}: {exc}",
                    level="warning",
                    tag="file",
                    label="TERMS",
                )

        matcher = _build_term_matcher(list(kb_entries.values()))
        data = {"entries": kb_entries, "matcher": matcher}
        _KB_INDEX_CACHE["key"] = cache_key
        _KB_INDEX_CACHE["data"] = data
        return data


def _ensure_kb_matcher(kb_data: dict[str, Any]) -> dict[str, Any]:
    matcher = kb_data.get("matcher")
    if matcher is None:
        entries = list((kb_data.get("entries") or {}).values())
        matcher = _build_term_matcher(entries)
        kb_data["matcher"] = matcher
    return matcher


def _retrieve_rare_token_candidates(
    text: str, kb_data: dict[str, Any], exclude_srcs: set[str]
) -> list[dict[str, Any]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return []
    matcher = _ensure_kb_matcher(kb_data)
    rare_inv = matcher.get("rare_token_inv", {})
    if not rare_inv:
        return []
    _, stems, _ = _tokenize_surface(text)
    out: list[dict[str, Any]] = []
    seen = set(exclude_srcs)
    for st in stems:
        if len(st) < 4 or st in _KB_STOP_TOKENS:
            continue
        for entry in rare_inv.get(st, ()):
            sl = entry["src"].lower()
            if sl not in seen:
                seen.add(sl)
                out.append(entry)
    return out


def _scan_text_for_kb_terms(text: str, kb_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return []
    matcher = _ensure_kb_matcher(kb_data)
    hits = _match_terms_in_text(text, matcher)
    return [
        {
            "src": hit.entry["src"],
            "tgt": hit.entry["tgt"],
            "note": str(hit.entry.get("note") or ""),
            "aliases": list(hit.matched_aliases),
            "src_matched": hit.src_matched,
            "fix_source": bool(hit.entry.get("fix_source", False)),
            "source": str(hit.entry.get("source") or "domain_kb"),
        }
        for hit in hits
    ]


def _weighted_char_len(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    return cjk * 2 + (len(text) - cjk)


def split_summary_chunks(max_weighted_chars: int | None = None) -> list[str]:
    limit = max_weighted_chars or SUMMARY_CHUNK_WEIGHTED_CHARS
    ws = get_active_workspace()
    utts = ws.load_utterances()
    lines = [
        str(u.get("src_punct") or u.get("src") or "").strip()
        for u in utts
        if str(u.get("src_punct") or u.get("src") or "").strip()
    ]
    if not lines:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        wlen = _weighted_char_len(line)
        if buf and buf_len + wlen > limit:
            chunks.append("\n".join(buf))
            buf = [line]
            buf_len = wlen
        else:
            buf.append(line)
            buf_len += wlen
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def combine_chunks() -> str:
    return "\n\n".join(split_summary_chunks())


def _valid_summary(response_data: Any) -> dict[str, str]:
    if not isinstance(response_data, dict):
        return {"status": "error", "message": "Response must be a JSON object"}
    terms = response_data.get("terms")
    if not isinstance(terms, list):
        return {"status": "error", "message": "Missing 'terms' list in JSON response"}
    for item in terms:
        if not isinstance(item, dict):
            return {"status": "error", "message": "Each term item must be an object"}
        src = str(item.get("src") or "").strip()
        tgt = str(item.get("tgt") or "").strip()
        if not src or not tgt:
            return {
                "status": "error",
                "message": "Term 'src' and 'tgt' must be non-empty",
            }
    return {"status": "success", "message": "Summary response validated"}


def _extract_chunk_terms(
    chunk_idx: int,
    chunk_text: str,
    chunk_kb_terms: list[dict[str, Any]],
    video_context_terms: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    check_cancel()
    prompt = get_summary_prompt(
        source_content=chunk_text,
        custom_terms_json={"terms": chunk_kb_terms},
        previously_extracted_terms=video_context_terms,
    )
    resp = ask_gpt(
        prompt,
        resp_type="json",
        valid_def=_valid_summary,
        log_title="summary",
    )
    if not isinstance(resp, dict):
        resp = {"theme": "", "terms": []}
    return chunk_idx, resp


def _merge_terminology_results(
    chunk_results: list[tuple[int, dict[str, Any]]],
    custom_terms_list: list[dict[str, Any]],
    kb_data: dict[str, Any] | None = None,
    full_text: str = "",
) -> dict[str, Any]:
    ordered_chunks = sorted(chunk_results, key=lambda x: x[0])
    themes = [
        str(res.get("theme") or res.get("topic") or "").strip()
        for _, res in ordered_chunks
        if isinstance(res, dict)
        and str(res.get("theme") or res.get("topic") or "").strip()
    ]
    combined_theme = " ".join(themes[:2]).strip()

    auth_by_stem: dict[tuple[str, ...], dict[str, Any]] = {}
    auth_by_lower: dict[str, dict[str, Any]] = {}
    for ct in custom_terms_list:
        s = str(ct.get("src") or "").strip()
        t = str(ct.get("tgt") or "").strip()
        if not s or not t:
            continue
        entry = {
            "src": s,
            "tgt": t,
            "note": str(ct.get("note") or "").strip(),
            "aliases": _normalize_aliases(ct.get("aliases"), s),
            "fix_source": bool(ct.get("fix_source", False)),
            "source": str(ct.get("source") or "custom"),
        }
        auth_by_lower[s.lower()] = entry
        _, st, _ = _tokenize_surface(s)
        if st:
            auth_by_stem.setdefault(tuple(st), entry)
        for al in entry["aliases"]:
            auth_by_lower.setdefault(al.lower(), entry)
            _, al_st, _ = _tokenize_surface(al)
            if al_st:
                auth_by_stem.setdefault(tuple(al_st), entry)

    if kb_data and kb_data.get("entries"):
        for k, entry in kb_data["entries"].items():
            auth_by_lower.setdefault(k, entry)
            _, st, _ = _tokenize_surface(entry["src"])
            if st:
                auth_by_stem.setdefault(tuple(st), entry)
            for al in entry.get("aliases") or []:
                al_str = str(al).strip()
                if not al_str:
                    continue
                auth_by_lower.setdefault(al_str.lower(), entry)
                _, al_st, _ = _tokenize_surface(al_str)
                if al_st:
                    auth_by_stem.setdefault(tuple(al_st), entry)

    scanned_aliases_by_src: dict[str, list[str]] = {}
    if kb_data and full_text:
        for scanned in _scan_text_for_kb_terms(full_text, kb_data):
            scanned_aliases_by_src[scanned["src"].lower()] = list(
                scanned.get("aliases") or []
            )

    merged_by_src: dict[str, dict[str, Any]] = {}
    stem_to_src_key: dict[tuple[str, ...], str] = {}

    def _upsert_term(
        src: str,
        tgt: str,
        note: str = "",
        aliases: list[str] | None = None,
        fix_source: bool = False,
        source: str = "task",
        overwrite_tgt: bool = False,
    ) -> None:
        s = src.strip()
        t = tgt.strip()
        if not s or not t:
            return
        _, st, _ = _tokenize_surface(s)
        st_tup = tuple(st) if st else ()

        auth = auth_by_lower.get(s.lower()) or (
            auth_by_stem.get(st_tup) if st_tup else None
        )
        extra_aliases = list(aliases or [])
        if auth is not None:
            if s.lower() != auth["src"].lower():
                extra_aliases.insert(0, s)
            s = auth["src"]
            t = auth["tgt"]
            if not note:
                note = str(auth.get("note") or "")
            fix_source = bool(auth.get("fix_source", False)) or fix_source
            auth_aliases = (
                scanned_aliases_by_src.get(s.lower(), [])
                if (kb_data and full_text and auth.get("source") != "custom")
                else (auth.get("aliases") or [])
            )
            extra_aliases = [*auth_aliases, *extra_aliases]
            _, st, _ = _tokenize_surface(s)
            st_tup = tuple(st) if st else ()

        key = s.lower()
        if key not in merged_by_src and st_tup and st_tup in stem_to_src_key:
            prev_key = stem_to_src_key[st_tup]
            t = merged_by_src[prev_key]["tgt"]

        clean_aliases = _normalize_aliases(extra_aliases, s)
        if key not in merged_by_src:
            merged_by_src[key] = {
                "src": s,
                "tgt": t,
                "note": note,
                "aliases": clean_aliases,
                "fix_source": fix_source,
                "source": source,
            }
            if st_tup:
                stem_to_src_key[st_tup] = key
        else:
            cur = merged_by_src[key]
            if overwrite_tgt:
                cur["tgt"] = t
            if note and not cur.get("note"):
                cur["note"] = note
            cur["fix_source"] = bool(cur.get("fix_source", False) or fix_source)
            cur["aliases"] = _normalize_aliases(
                [*(cur.get("aliases") or []), *clean_aliases], cur["src"]
            )
            if source == "task":
                cur["source"] = "task"

    for ct in custom_terms_list:
        if ct.get("source") == "custom":
            _upsert_term(
                ct["src"],
                ct["tgt"],
                str(ct.get("note") or ""),
                ct.get("aliases") or [],
                bool(ct.get("fix_source", False)),
                source="custom",
                overwrite_tgt=True,
            )

    for _, res in ordered_chunks:
        if not isinstance(res, dict):
            continue
        for item in res.get("terms") or []:
            if not isinstance(item, dict):
                continue
            s = str(item.get("src") or "").strip()
            t = str(item.get("tgt") or "").strip()
            n_val = item.get("note")
            n_str = (
                ""
                if n_val is None or str(n_val).lower() == "nan"
                else str(n_val).strip()
            )
            als = _normalize_aliases(item.get("aliases"), s)
            _upsert_term(s, t, n_str, als, source="task")

    return {"theme": combined_theme, "terms": list(merged_by_src.values())}


@check_file_exists("summarize")
def get_summary() -> None:
    ws = get_active_workspace()
    custom_terms_list = load_custom_terms()
    kb_data = _get_kb_index(custom_terms_list)
    text_chunks = split_summary_chunks()
    full_text = "\n".join(text_chunks)

    if not text_chunks:
        ws.save_terms("", [])
        return

    video_kb_terms = _scan_text_for_kb_terms(full_text, kb_data)
    video_kb_srcs = {t["src"].lower() for t in video_kb_terms}
    total_chunks = len(text_chunks)
    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))

    report_progress(
        0.1,
        f"术语提取 0/{total_chunks} 批",
        f"开始并行提取术语与摘要（共 {total_chunks} 批，全片预命中 {len(video_kb_terms)} 条权威术语）",
        tag="llm",
        label="TERMS",
    )

    chunk_results: list[tuple[int, dict[str, Any]]] = []
    with ContextThreadPoolExecutor(
        max_workers=min(max_workers, total_chunks)
    ) as executor:
        futures = []
        for idx, chunk in enumerate(text_chunks):
            chunk_kb = _scan_text_for_kb_terms(chunk, kb_data)
            chunk_srcs = {t["src"].lower() for t in chunk_kb}
            rare_cands = _retrieve_rare_token_candidates(
                chunk, kb_data, chunk_srcs | video_kb_srcs
            )
            context_kb = [
                *(t for t in video_kb_terms if t["src"].lower() not in chunk_srcs),
                *rare_cands,
            ]
            futures.append(
                executor.submit(_extract_chunk_terms, idx, chunk, chunk_kb, context_kb)
            )
        for done_count, fut in enumerate(futures, 1):
            check_cancel()
            res = fut.result()
            chunk_results.append(res)
            report_progress(
                0.1 + 0.8 * (done_count / total_chunks),
                f"术语提取 {done_count}/{total_chunks} 批",
                f"已完成第 {done_count}/{total_chunks} 批术语提取",
                tag="llm",
                label="TERMS",
            )

    merged = _merge_terminology_results(
        chunk_results, custom_terms_list, kb_data=kb_data, full_text=full_text
    )
    ws.save_terms(merged.get("theme", ""), merged.get("terms", []))
    report_progress(
        1.0,
        f"术语提取完成（共 {len(merged.get('terms', []))} 条）",
        f"术语提取完成，共汇总 {len(merged.get('terms', []))} 条术语",
        tag="llm",
        label="TERMS",
    )


_TERM_INDEX_LOCK = threading.Lock()
_TERM_INDEX_CACHE: dict[str, Any] = {"key": None, "data": None}


def _get_term_index() -> dict[str, Any]:
    ws = get_active_workspace()
    active_domain = _active_glossary_domain()
    dom_csv = domain_terms_csv_path(active_domain) if active_domain else Path("")
    dom_dict = domain_dict_json_path(active_domain) if active_domain else Path("")
    ws_mtime = ws.get_terms_mtime() if ws.has_terms() else 0.0
    ext_mtime = max(
        _file_mtime(CUSTOM_TERMS_PATH),
        _file_mtime(OPTIONAL_CSV_TERMS_PATH),
        _file_mtime(OPTIONAL_JSON_TERMS_PATH),
        _file_mtime(GLOBAL_GLOSSARY_PATH),
        _file_mtime(dom_csv) if active_domain else 0.0,
        _file_mtime(dom_dict) if active_domain else 0.0,
    )
    cache_key = (str(ws.root), active_domain, ws_mtime, ext_mtime)

    with _TERM_INDEX_LOCK:
        if (
            _TERM_INDEX_CACHE["key"] == cache_key
            and _TERM_INDEX_CACHE["data"] is not None
        ):
            return _TERM_INDEX_CACHE["data"]

        task_terms = (ws.load_terms().get("terms") or []) if ws.has_terms() else []
        custom_terms = load_custom_terms()
        kb_data = _get_kb_index(custom_terms)

        primary = task_terms if ws_mtime >= ext_mtime else custom_terms
        secondary = custom_terms if ws_mtime >= ext_mtime else task_terms

        merged_map: dict[str, dict[str, Any]] = {}
        for item in (*primary, *secondary):
            if not isinstance(item, dict):
                continue
            s = str(item.get("src") or "").strip()
            t = str(item.get("tgt") or "").strip()
            if not s or not t:
                continue
            k = s.lower()
            als = _normalize_aliases(item.get("aliases"), s)
            if k not in merged_map:
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

        all_entries = list(merged_map.values())
        for k, kb_entry in (kb_data.get("entries") or {}).items():
            if k not in merged_map:
                all_entries.append(kb_entry)

        matcher = _build_term_matcher(all_entries)
        data = {"terms": list(merged_map.values()), "matcher": matcher}
        _TERM_INDEX_CACHE["key"] = cache_key
        _TERM_INDEX_CACHE["data"] = data
        return data


def search_things_to_note_in_prompt(sentence: str) -> str | None:
    if not sentence or not sentence.strip():
        return None
    idx_data = _get_term_index()
    hits = _match_terms_in_text(sentence, idx_data["matcher"])
    lines = format_term_hits_for_prompt(hits)
    return "\n".join(lines) if lines else None

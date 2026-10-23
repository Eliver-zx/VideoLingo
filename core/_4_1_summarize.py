from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
from difflib import SequenceMatcher

import pandas as pd

from core.prompts import get_summary_prompt
from core.utils import (
    ask_gpt,
    check_cancel,
    check_file_exists,
    load_key_or,
    rprint,
)
from core.utils.models import _3_2_SPLIT_BY_MEANING, _4_1_TERMINOLOGY, resolve_task_path
from core.workspace import (
    GLOBAL_GLOSSARY_CSV,
    TaskWorkspace,
    get_active_workspace,
    load_global_glossary,
)

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore[import-not-found]
except ImportError:
    _rf_fuzz = None

CUSTOM_TERMS_PATH = "custom_terms.xlsx"
OPTIONAL_CSV_TERMS_PATH = "Term.csv"
OPTIONAL_JSON_TERMS_PATH = "term-map.json"
GLOBAL_GLOSSARY_PATH = str(GLOBAL_GLOSSARY_CSV)

# ~15 minutes of speech: ~2,500 English words (~14,000 ASCII chars at weight 1.0)
# or ~4,000 Chinese characters (at weight 3.5 -> 14,000 weighted chars).
SUMMARY_CHUNK_WEIGHTED_CHARS = 14000
FUZZY_THRESHOLD = 82.0

_INDEX_LOCK = threading.Lock()
_CACHED_INDEX_KEY: tuple | None = None
_CACHED_INDEX_DATA: dict | None = None

_TOKEN_RE = re.compile(
    r"[a-z0-9]+|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", re.IGNORECASE
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

_INVARIANT_WORDS = {
    "series",
    "species",
    "corpus",
    "status",
    "basis",
    "axis",
    "crisis",
    "thesis",
    "analysis",
    "diagnosis",
    "synopsis",
    "hypothesis",
    "parenthesis",
    "oasis",
    "chassis",
    "debris",
    "tennis",
    "pelvis",
    "iris",
    "epidermis",
    "physics",
    "graphics",
    "mathematics",
    "statistics",
    "dynamics",
    "kinematics",
    "acoustics",
    "optics",
    "robotics",
    "electronics",
    "mechanics",
    "economics",
    "politics",
    "ethics",
    "linguistics",
    "logistics",
    "genetics",
    "athletics",
    "canvas",
    "alias",
    "atlas",
    "bias",
    "compass",
    "gas",
    "lens",
    "bus",
    "plus",
    "minus",
    "virus",
    "bonus",
    "campus",
    "census",
    "chorus",
    "circus",
    "consensus",
    "focus",
    "genius",
    "nexus",
    "radius",
    "sinus",
    "stimulus",
    "surplus",
    "syllabus",
    "terminus",
    "apparatus",
    "prospectus",
    "string",
    "spring",
    "bring",
    "thing",
    "king",
    "ring",
    "wing",
    "sing",
    "cling",
    "fling",
    "sling",
    "sting",
    "swing",
    "wring",
    "ceiling",
    "feeling",
    "morning",
    "evening",
    "warning",
    "lightning",
    "nothing",
    "anything",
    "everything",
    "something",
    "during",
    "sterling",
    "darling",
    "sibling",
    "seed",
    "feed",
    "need",
    "weed",
    "speed",
    "bleed",
    "breed",
    "freed",
    "greed",
    "steed",
    "creed",
    "deed",
    "heed",
    "reed",
    "proceed",
    "succeed",
    "exceed",
    "indeed",
    "red",
    "bed",
    "led",
    "fed",
    "wed",
    "shed",
    "sled",
    "shred",
    "thread",
    "bread",
    "dead",
    "head",
    "lead",
    "read",
    "spread",
    "tread",
    "dread",
    "instead",
    "ahead",
    "news",
    "means",
    "goods",
    "odds",
    "hq",
    "os",
    "ios",
    "macos",
    "windows",
    "kubernetes",
    "postgres",
    "redis",
    "aws",
    "devops",
}


def _stem_en_token(token: str) -> str:
    """Lightweight English plural/suffix normalization with invariant word protection."""
    t = token.lower()
    if len(t) <= 3 or not t.isascii() or not t.isalpha() or t in _INVARIANT_WORDS:
        return t
    if t.endswith("ies") and len(t) > 5:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        if t.endswith(("ches", "shes", "xes", "zes", "sses")):
            return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is", "as", "os", "ys")):
        return t[:-1]
    if t.endswith("ing") and len(t) >= 7:
        base = t[:-3]
        if base in _INVARIANT_WORDS:
            return base
        if len(base) >= 4 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        if len(base) >= 4:
            return base
    return t


def _tokenize_and_normalize(text: str) -> tuple[list[str], list[str]]:
    """Return ``(raw_lower_tokens, stemmed_tokens)`` for a text segment."""
    raw = [m.group(0).lower() for m in _TOKEN_RE.finditer(str(text))]
    stemmed = [_stem_en_token(tok) for tok in raw]
    return raw, stemmed


def _similarity_score(a: str, b: str) -> float:
    """Return similarity score in [0, 100] between two normalized strings."""
    if a == b:
        return 100.0
    if _rf_fuzz is not None:
        return float(_rf_fuzz.ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _is_Bounded_fuzzy_match(norm_term: str, cand: str, k: int) -> bool:
    """Determine if ``cand`` is a genuine ASR phonetic/spelling variant of ``norm_term``.

    Prevents common English lookalikes (e.g., ``faster`` vs ``raster``, ``severe`` vs ``server``,
    ``mode`` vs ``model``) from firing false positives on single-word terms:
    - Single-word terms (``k == 1``) must have length >= 5, share the same first character,
      and either have similarity >= 88.0 (for len >= 7) or match a known phonetic suffix/vowel
      substitution (like ``bungie`` <-> ``bungee`` / ``banji``, ``raycasting`` <-> ``raycastin``).
    - Multi-word terms (``k >= 2``) have high context specificity and use ``FUZZY_THRESHOLD`` (82.0).
    """
    if norm_term == cand:
        return True
    score = _similarity_score(norm_term, cand)
    if k >= 2:
        return score >= FUZZY_THRESHOLD

    # Single-word term (k == 1): strict guards against everyday English lookalikes
    if len(norm_term) < 5 or len(cand) < 5:
        return False
    if norm_term[0] != cand[0]:
        return False
    # Reject 1-char prefix/suffix extensions like "mode" <-> "model" unless trailing 'g' drop ("raycastin")
    if norm_term.startswith(cand) or cand.startswith(norm_term):
        longer, shorter = (
            (norm_term, cand) if len(norm_term) > len(cand) else (cand, norm_term)
        )
        tail = longer[len(shorter) :]
        if tail in ("g", "e", "y", "ie", "ee") and len(shorter) >= 5:
            return score >= FUZZY_THRESHOLD
        return False

    if len(norm_term) >= 8:
        return score >= 86.0
    # For 5..7 char single words, require phonetic equivalence on the differing part
    # (e.g. bungie <-> bungee / bungy) or score >= 90.0
    phonetic_norm_a = (
        norm_term.replace("ie", "ee").replace("y", "ee").replace("ph", "f")
    )
    phonetic_norm_b = cand.replace("ie", "ee").replace("y", "ee").replace("ph", "f")
    if phonetic_norm_a == phonetic_norm_b:
        return True
    return score >= 90.0


def _weighted_char_len(text: str) -> float:
    """Calculate weighted character length where CJK characters count as 3.5 and others as 1.0."""
    total = 0.0
    for ch in str(text):
        if (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        ):
            total += 3.5
        else:
            total += 1.0
    return total


def split_summary_chunks(
    sentences: list[str],
    max_weighted_chars: float = SUMMARY_CHUNK_WEIGHTED_CHARS,
) -> list[str]:
    """Group sentences into ~15-minute chunks (~14,000 weighted chars each)."""
    chunks: list[str] = []
    current_lines: list[str] = []
    current_weight = 0.0

    for line in sentences:
        stripped = line.strip()
        if not stripped:
            continue
        w = _weighted_char_len(stripped) + 1.0
        if current_lines and current_weight + w > max_weighted_chars:
            chunks.append(" ".join(current_lines))
            current_lines = [stripped]
            current_weight = w
        else:
            current_lines.append(stripped)
            current_weight += w

    if current_lines:
        chunks.append(" ".join(current_lines))
    return chunks


def combine_chunks() -> str:
    """Combine sentence lines from ``_3_2_SPLIT_BY_MEANING`` into a single text string."""
    meaning_path = resolve_task_path(_3_2_SPLIT_BY_MEANING)
    with open(meaning_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]
    combined_text = " ".join(sentences)
    limit = load_key_or("summary_length", None)
    if isinstance(limit, int) and limit > 0:
        return combined_text[:limit]
    return combined_text


def _normalize_aliases(raw_aliases: Any, src: str = "") -> list[str]:
    if not raw_aliases:
        return []
    if isinstance(raw_aliases, str):
        items = [x.strip() for x in re.split(r"[,;|/，；｜]", raw_aliases) if x.strip()]
    elif isinstance(raw_aliases, (list, tuple, set)):
        items = [str(x).strip() for x in raw_aliases if str(x).strip()]
    else:
        return []
    src_low = src.strip().lower()
    return list(dict.fromkeys(a for a in items if a.lower() != src_low and a.lower() != "nan"))


def load_custom_terms() -> list[dict[str, Any]]:
    """Load user-defined glossary terms from ``glossary/custom_terms.csv``, ``custom_terms.xlsx``,
    and optional local term files.

    Precedence (highest first): ``glossary/custom_terms.csv`` / ``custom_terms.xlsx`` > ``Term.csv`` > ``term-map.json``.
    """
    merged: dict[str, dict[str, Any]] = {}

    def _add_term(
        src: object,
        tgt: object,
        note: object = "",
        aliases: Any = None,
        fix_source: bool = False,
        source: str = "custom",
    ) -> None:
        s = str(src).strip()
        t = str(tgt).strip()
        n = "" if (note is None or note != note) else str(note).strip()
        if not s or not t or s.lower() == "nan" or t.lower() == "nan":
            return
        alias_list = _normalize_aliases(aliases, s)
        key = s.lower()
        if key not in merged:
            merged[key] = {
                "src": s,
                "tgt": t,
                "note": n,
                "aliases": alias_list,
                "fix_source": bool(fix_source),
                "source": source,
            }
        else:
            existing_aliases = merged[key].get("aliases") or []
            merged[key]["aliases"] = list(dict.fromkeys([*existing_aliases, *alias_list]))
            if not merged[key].get("note") and n:
                merged[key]["note"] = n

    use_global = (
        GLOBAL_GLOSSARY_PATH != str(GLOBAL_GLOSSARY_CSV)
        or CUSTOM_TERMS_PATH == "custom_terms.xlsx"
    )
    if use_global and os.path.exists(GLOBAL_GLOSSARY_PATH):
        try:
            for item in load_global_glossary(GLOBAL_GLOSSARY_PATH):
                _add_term(
                    item.get("src", ""),
                    item.get("tgt", ""),
                    item.get("note", ""),
                    aliases=item.get("aliases"),
                    fix_source=bool(item.get("fix_source", False)),
                    source="global_kb",
                )
        except Exception as exc:
            rprint(f"[yellow]⚠️ Could not read {GLOBAL_GLOSSARY_PATH}: {exc}[/yellow]")

    if os.path.exists(CUSTOM_TERMS_PATH):
        try:
            df = pd.read_excel(CUSTOM_TERMS_PATH)
            for _, row in df.iterrows():
                if len(row) >= 2:
                    note_val = row.iloc[2] if len(row) >= 3 else ""
                    aliases_val = row.get("aliases", "") if hasattr(row, "get") else ""
                    _add_term(row.iloc[0], row.iloc[1], note_val, aliases=aliases_val)
        except Exception as exc:
            rprint(f"[yellow]⚠️ Could not read {CUSTOM_TERMS_PATH}: {exc}[/yellow]")

    if os.path.exists(OPTIONAL_CSV_TERMS_PATH):
        try:
            df_csv = pd.read_csv(
                OPTIONAL_CSV_TERMS_PATH, header=None, names=["src", "tgt", "note"]
            )
            for _, row in df_csv.iterrows():
                if str(row["src"]).strip().lower() in ("src", "source", "term"):
                    continue
                _add_term(row["src"], row["tgt"], row.get("note", ""))
        except Exception as exc:
            rprint(
                f"[yellow]⚠️ Could not read {OPTIONAL_CSV_TERMS_PATH}: {exc}[/yellow]"
            )

    if os.path.exists(OPTIONAL_JSON_TERMS_PATH):
        try:
            with open(OPTIONAL_JSON_TERMS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str):
                        _add_term(k, v, "")
                    elif isinstance(v, dict):
                        _add_term(
                            k,
                            v.get("tgt", ""),
                            v.get("note", ""),
                            aliases=v.get("aliases"),
                        )
        except Exception as exc:
            rprint(
                f"[yellow]⚠️ Could not read {OPTIONAL_JSON_TERMS_PATH}: {exc}[/yellow]"
            )

    return list(merged.values())


def _file_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _build_term_index(terms: list[dict[str, Any]]) -> dict:
    """Build fast lookup structures for exact, stemmed, alias, and N-gram fuzzy matching."""
    cjk_or_symbol_map: dict[str, dict[str, Any]] = {}
    word_boundary_res: list[tuple[re.Pattern[str], dict[str, Any]]] = []
    stem_ngram_map: dict[tuple[str, ...], dict[str, Any]] = {}
    fuzzy_entries: list[tuple[str, int, dict[str, Any]]] = []
    by_k: dict[int, list[dict[str, Any]]] = {}
    max_k = 1

    for term in terms:
        src = str(term.get("src", "")).strip()
        if not src:
            continue
        tgt = str(term.get("tgt", "")).strip()
        note = str(term.get("note", "")).strip()
        aliases = _normalize_aliases(term.get("aliases"), src)
        entry: dict[str, Any] = {
            "src": src,
            "tgt": tgt,
            "note": note,
            "aliases": aliases,
            "fix_source": bool(term.get("fix_source", False)),
            "source": str(term.get("source") or "task"),
        }

        for surface in [src, *aliases]:
            surf_lower = surface.strip().lower()
            if not surf_lower:
                continue

            # Substring matching is ONLY safe for CJK terms; ASCII/Latin terms require word boundaries
            if _CJK_RE.search(surf_lower):
                if surf_lower not in cjk_or_symbol_map:
                    cjk_or_symbol_map[surf_lower] = entry
            elif re.search(r"[^a-z0-9\s]", surf_lower):
                # Terms with symbols like "C++", "C#", ".NET", "x86_64"
                pattern = re.compile(
                    rf"(?<![a-z0-9]){re.escape(surf_lower)}(?![a-z0-9])", re.IGNORECASE
                )
                word_boundary_res.append((pattern, entry))

            raw_toks, stem_toks = _tokenize_and_normalize(surface)
            if stem_toks:
                k = len(stem_toks)
                if k > max_k:
                    max_k = k
                key_tuple = tuple(stem_toks)
                if key_tuple not in stem_ngram_map:
                    stem_ngram_map[key_tuple] = entry
                norm_str = " ".join(raw_toks)
                by_k.setdefault(k, []).append(
                    {
                        "surface": surface,
                        "norm": norm_str,
                        "stem_tokens": key_tuple,
                        "entry": entry,
                    }
                )
                if len(norm_str) >= 5 or (k >= 2 and len(norm_str) >= 4):
                    fuzzy_entries.append((norm_str, k, entry))

    return {
        "terms": terms,
        "by_k": by_k,
        "cjk_or_symbol_map": cjk_or_symbol_map,
        "word_boundary_res": word_boundary_res,
        "stem_ngram_map": stem_ngram_map,
        "fuzzy_entries": fuzzy_entries,
        "max_k": min(max_k, 8),
    }


def _get_term_index() -> dict:
    """Return the cached term index, rebuilding automatically if any source file changed."""
    global _CACHED_INDEX_KEY, _CACHED_INDEX_DATA

    term_json_path = resolve_task_path(_4_1_TERMINOLOGY)
    terms_contract_path = str(Path(term_json_path).with_name("terms.json"))
    abs_json = os.path.abspath(term_json_path)
    abs_terms_contract = os.path.abspath(terms_contract_path)
    abs_global = os.path.abspath(GLOBAL_GLOSSARY_PATH)
    abs_xlsx = os.path.abspath(CUSTOM_TERMS_PATH)
    abs_csv = os.path.abspath(OPTIONAL_CSV_TERMS_PATH)
    abs_map = os.path.abspath(OPTIONAL_JSON_TERMS_PATH)
    json_mtime = _file_mtime(abs_json)
    contract_mtime = _file_mtime(abs_terms_contract)
    global_mtime = _file_mtime(abs_global)
    xlsx_mtime = _file_mtime(abs_xlsx)
    csv_mtime = _file_mtime(abs_csv)
    map_mtime = _file_mtime(abs_map)

    cache_key = (
        abs_json,
        json_mtime,
        abs_terms_contract,
        contract_mtime,
        abs_global,
        global_mtime,
        abs_xlsx,
        xlsx_mtime,
        abs_csv,
        csv_mtime,
        abs_map,
        map_mtime,
    )

    with _INDEX_LOCK:
        if _CACHED_INDEX_KEY == cache_key and _CACHED_INDEX_DATA is not None:
            return _CACHED_INDEX_DATA

        merged_by_src: dict[str, dict[str, Any]] = {}

        def _merge_json_terms(file_path: str) -> None:
            if not os.path.exists(file_path):
                return
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return
            for item in data.get("terms", []) if isinstance(data, dict) else []:
                if isinstance(item, dict) and item.get("src"):
                    src_str = str(item.get("src", "")).strip()
                    key = src_str.lower()
                    prev = merged_by_src.get(key, {})
                    prev_aliases = prev.get("aliases") or []
                    new_aliases = _normalize_aliases(item.get("aliases"), src_str)
                    merged_by_src[key] = {
                        "src": src_str,
                        "tgt": str(item.get("tgt", "")).strip(),
                        "note": str(item.get("note", "") or prev.get("note", "")).strip(),
                        "aliases": list(dict.fromkeys([*prev_aliases, *new_aliases])),
                        "fix_source": bool(item.get("fix_source", prev.get("fix_source", False))),
                        "source": str(item.get("source") or prev.get("source") or "task"),
                    }

        custom_list = load_custom_terms()
        custom_mtimes = [m for m in (global_mtime, xlsx_mtime, csv_mtime, map_mtime) if m is not None]
        latest_custom_mtime = max(custom_mtimes) if custom_mtimes else 0.0
        task_mtimes = [m for m in (json_mtime, contract_mtime) if m is not None]
        latest_task_mtime = max(task_mtimes) if task_mtimes else 0.0
        json_is_newer = bool(task_mtimes) and latest_task_mtime > latest_custom_mtime

        if not json_is_newer:
            _merge_json_terms(terms_contract_path)
            _merge_json_terms(term_json_path)
            for item in custom_list:
                key = item["src"].lower()
                prev = merged_by_src.get(key, {})
                merged_by_src[key] = {
                    **item,
                    "aliases": list(dict.fromkeys([*(prev.get("aliases") or []), *(item.get("aliases") or [])])),
                }
        else:
            for item in custom_list:
                merged_by_src[item["src"].lower()] = dict(item)
            if contract_mtime is not None and (json_mtime is None or contract_mtime >= json_mtime):
                _merge_json_terms(term_json_path)
                _merge_json_terms(terms_contract_path)
            else:
                _merge_json_terms(terms_contract_path)
                _merge_json_terms(term_json_path)

        index_data = _build_term_index(list(merged_by_src.values()))
        _CACHED_INDEX_KEY = cache_key
        _CACHED_INDEX_DATA = index_data
        return index_data


def search_things_to_note_in_prompt(sentence: str) -> str | None:
    """Search for matching terminology in ``sentence`` via exact, lemma/stem, and N-gram fuzzy matching."""
    if not sentence or not str(sentence).strip():
        return None

    index_data = _get_term_index()
    terms = index_data["terms"]
    if not terms:
        return None

    sentence_lower = str(sentence).lower()
    raw_toks, stem_toks = _tokenize_and_normalize(sentence_lower)
    n_toks = len(raw_toks)
    max_k = index_data["max_k"]

    matched_srcs: set[str] = set()

    # 1. Substring check ONLY for CJK terms, plus bounded regex for symbol terms (e.g. C++, .NET)
    for src_lower, entry in index_data["cjk_or_symbol_map"].items():
        if src_lower in sentence_lower:
            matched_srcs.add(entry["src"])
    for pattern, entry in index_data["word_boundary_res"]:
        if pattern.search(sentence_lower):
            matched_srcs.add(entry["src"])

    # 2. Exact & stemmed token N-gram hash lookup (O(N * K)) — respects exact token boundaries
    stem_ngram_map = index_data["stem_ngram_map"]
    raw_ngrams_by_k: dict[int, set[str]] = {}
    for k in range(1, min(max_k, n_toks) + 1):
        raw_set: set[str] = set()
        for i in range(n_toks - k + 1):
            stem_gram = tuple(stem_toks[i : i + k])
            if stem_gram in stem_ngram_map:
                matched_srcs.add(stem_ngram_map[stem_gram]["src"])
            raw_set.add(" ".join(raw_toks[i : i + k]))
        raw_ngrams_by_k[k] = raw_set

    # 3. Word-level N-gram fuzzy matching for ASR spelling/phonetic variants
    for norm_term, k, entry in index_data["fuzzy_entries"]:
        if entry["src"] in matched_srcs:
            continue
        candidate_ngrams = raw_ngrams_by_k.get(k)
        if not candidate_ngrams:
            continue
        term_len = len(norm_term)
        for cand in candidate_ngrams:
            if abs(len(cand) - term_len) > max(2, int(term_len * 0.28)):
                continue
            if _is_Bounded_fuzzy_match(norm_term, cand, k):
                matched_srcs.add(entry["src"])
                break

    if not matched_srcs:
        return None

    selected = [t for t in terms if t["src"] in matched_srcs]
    if not selected:
        return None

    return "\n".join(
        f'{i + 1}. "{term["src"]}": "{term["tgt"]}", meaning: {term["note"]}'
        for i, term in enumerate(selected)
    )


def _valid_summary(response_data):
    required_keys = {"src", "tgt", "note"}
    if (
        not isinstance(response_data, dict)
        or "terms" not in response_data
        or not isinstance(response_data["terms"], list)
    ):
        return {"status": "error", "message": "Invalid response format"}
    for term in response_data["terms"]:
        if not isinstance(term, dict) or not all(key in term for key in required_keys):
            return {"status": "error", "message": "Invalid response format"}
    return {"status": "success", "message": "Summary completed"}


def _extract_chunk_summary(
    chunk_idx: int, chunk_text: str, custom_terms_json: dict
) -> tuple[int, dict]:
    check_cancel()
    prompt = get_summary_prompt(chunk_text, custom_terms_json)
    log_title = "summary" if chunk_idx == 0 else f"summary_{chunk_idx}"
    resp = ask_gpt(
        prompt, resp_type="json", valid_def=_valid_summary, log_title=log_title
    )
    return chunk_idx, resp


def _merge_terminology_results(
    chunk_results: list[tuple[int, dict]],
    custom_terms_list: list[dict[str, Any]],
) -> dict:
    """Deterministically unify terminology across chunks and authoritative custom terms."""
    chunk_results_sorted = sorted(chunk_results, key=lambda x: x[0])

    themes: list[str] = []
    for _, res in chunk_results_sorted:
        t = str(res.get("theme") or res.get("topic") or "").strip()
        if t and t not in themes:
            themes.append(t)
    combined_theme = " ".join(themes[:2]) if themes else ""

    by_src_lower: dict[str, dict[str, Any]] = {}
    by_stem_key: dict[tuple[str, ...], str] = {}

    for term in custom_terms_list:
        src = str(term["src"]).strip()
        tgt = str(term["tgt"]).strip()
        note = str(term.get("note", "")).strip()
        if not src or not tgt:
            continue
        s_low = src.lower()
        aliases = _normalize_aliases(term.get("aliases"), src)
        entry: dict[str, Any] = {"src": src, "tgt": tgt, "note": note}
        if aliases:
            entry["aliases"] = aliases
        if term.get("fix_source"):
            entry["fix_source"] = True
        if term.get("source"):
            entry["source"] = str(term["source"])
        by_src_lower[s_low] = entry
        _, stems = _tokenize_and_normalize(src)
        if stems:
            by_stem_key[tuple(stems)] = tgt

    for _, res in chunk_results_sorted:
        for raw_term in res.get("terms", []):
            if not isinstance(raw_term, dict):
                continue
            src = str(raw_term.get("src", "")).strip()
            tgt = str(raw_term.get("tgt", "")).strip()
            note = str(raw_term.get("note", "")).strip()
            if not src or not tgt:
                continue
            s_low = src.lower()
            aliases = _normalize_aliases(raw_term.get("aliases"), src)
            _, stems = _tokenize_and_normalize(src)
            stem_tuple = tuple(stems) if stems else ()

            if s_low in by_src_lower:
                if not by_src_lower[s_low].get("note") and note:
                    by_src_lower[s_low]["note"] = note
                if aliases:
                    prev_a = by_src_lower[s_low].get("aliases") or []
                    by_src_lower[s_low]["aliases"] = list(dict.fromkeys([*prev_a, *aliases]))
                continue

            if stem_tuple and stem_tuple in by_stem_key:
                tgt = by_stem_key[stem_tuple]
            elif stem_tuple:
                by_stem_key[stem_tuple] = tgt

            entry = {"src": src, "tgt": tgt, "note": note}
            if aliases:
                entry["aliases"] = aliases
            by_src_lower[s_low] = entry

    return {
        "theme": combined_theme,
        "topic": combined_theme,
        "terms": list(by_src_lower.values()),
    }


# Kept on a retry, so that the edits of `pause_before_translate` are not overwritten
@check_file_exists(_4_1_TERMINOLOGY)
def get_summary():
    meaning_path = resolve_task_path(_3_2_SPLIT_BY_MEANING)
    term_json_path = resolve_task_path(_4_1_TERMINOLOGY)
    with open(meaning_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    custom_terms_list = load_custom_terms()
    custom_terms_json = {"terms": custom_terms_list}
    if custom_terms_list:
        rprint(f"📖 Custom Terms Loaded: {len(custom_terms_list)} terms")
        rprint(
            "📝 Terms Content:",
            json.dumps(custom_terms_json, indent=2, ensure_ascii=False),
        )

    chunks = split_summary_chunks(
        sentences, max_weighted_chars=SUMMARY_CHUNK_WEIGHTED_CHARS
    )
    if not chunks:
        src_content = combine_chunks()
        chunks = [src_content] if src_content else [""]

    rprint(
        f"📝 Summarizing and extracting terminology across {len(chunks)} chunk(s) ..."
    )

    chunk_results: list[tuple[int, dict]] = []
    if len(chunks) == 1:
        chunk_results.append(_extract_chunk_summary(0, chunks[0], custom_terms_json))
    else:
        max_workers = max(1, int(load_key_or("max_workers", 4) or 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _extract_chunk_summary, idx, chunk_text, custom_terms_json
                )
                for idx, chunk_text in enumerate(chunks)
            ]
            try:
                for future in concurrent.futures.as_completed(futures):
                    check_cancel()
                    chunk_results.append(future.result())
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

    summary = _merge_terminology_results(chunk_results, custom_terms_list)

    os.makedirs(os.path.dirname(term_json_path) or ".", exist_ok=True)
    with open(term_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    ws = (
        get_active_workspace()
        if str(_4_1_TERMINOLOGY).startswith("output/")
        else TaskWorkspace(
            Path(term_json_path).parent.parent
            if Path(term_json_path).parent.name == "log"
            else Path(term_json_path).parent
        )
    )
    ws.save_terms(summary.get("theme", ""), summary.get("terms", []))

    rprint(
        f"💾 Summary log saved to → `{term_json_path}` ({len(summary['terms'])} terms)"
    )


if __name__ == "__main__":
    get_summary()

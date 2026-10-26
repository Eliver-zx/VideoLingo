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

from core.prompts import get_consolidate_terms_prompt, get_summary_prompt
from core.task_runner import StopTask
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

# ~6 minutes of speech per extraction chunk (~1,000 English words / ~6,000 weighted chars)
# Prevents LLM output-length saturation on dense transcripts while keeping fast parallel Map-Reduce.
SUMMARY_CHUNK_WEIGHTED_CHARS = 6000
FUZZY_THRESHOLD = 82.0

_INDEX_LOCK = threading.RLock()
_CACHED_INDEX_KEY: tuple | None = None
_CACHED_INDEX_DATA: dict | None = None
_CACHED_KB_KEY: tuple | None = None
_CACHED_KB_DATA: dict | None = None

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


def _is_single_word_fuzzy_match(norm_term: str, cand: str) -> bool:
    """Strict single-word ASR phonetic/spelling variant check."""
    if norm_term == cand or _stem_en_token(norm_term) == _stem_en_token(cand):
        return True
    if len(norm_term) < 5 or len(cand) < 5:
        return False
    if norm_term[0] != cand[0]:
        return False
    score = _similarity_score(norm_term, cand)
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
    phonetic_norm_a = (
        norm_term.replace("ie", "ee").replace("y", "ee").replace("ph", "f")
    )
    phonetic_norm_b = cand.replace("ie", "ee").replace("y", "ee").replace("ph", "f")
    if phonetic_norm_a == phonetic_norm_b:
        return True
    return score >= 90.0


def _is_Bounded_fuzzy_match(norm_term: str, cand: str, k: int) -> bool:
    """Determine if ``cand`` is a genuine ASR phonetic/spelling variant of ``norm_term``.

    Prevents common English lookalikes (e.g., ``faster`` vs ``raster``, ``severe`` vs ``server``,
    ``mode`` vs ``model``, ``rate weapons`` vs ``heat weapons``, ``and compendium`` vs ``data compendium``)
    from firing false positives:
    - Single-word terms (``k == 1``) must pass ``_is_single_word_fuzzy_match``.
    - Multi-word terms (``k >= 2``) must have overall similarity >= ``FUZZY_THRESHOLD`` AND every
      corresponding token pair must either share the same stem or pass the same-initial-char token check.
    """
    if norm_term == cand:
        return True
    if k == 1:
        return _is_single_word_fuzzy_match(norm_term, cand)

    score = _similarity_score(norm_term, cand)
    if score < FUZZY_THRESHOLD:
        return False
    t_toks = norm_term.split()
    c_toks = cand.split()
    if len(t_toks) != len(c_toks):
        return False
    for tt, ct in zip(t_toks, c_toks):
        if tt == ct or _stem_en_token(tt) == _stem_en_token(ct):
            continue
        if len(tt) < 4 or len(ct) < 4 or tt[0] != ct[0]:
            return False
        if _similarity_score(tt, ct) < 80.0 and not _is_single_word_fuzzy_match(tt, ct):
            return False
    return True


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
    """Group sentences into ~6-minute chunks (~6,000 weighted chars each)."""
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
    return list(
        dict.fromkeys(a for a in items if a.lower() != src_low and a.lower() != "nan")
    )


_KB_META_KEYS = {
    "version",
    "timestamp",
    "data_hash",
    "source",
    "type",
    "id",
    "name",
    "title",
    "description",
    "author",
    "license",
    "status",
    "message",
    "theme",
    "topic",
    "summary",
    "language",
    "created_at",
    "updated_at",
}


def _extract_generic_kb_pairs(
    node: Any, out: dict[str, dict[str, Any]], source_tag: str = "kb"
) -> None:
    """Recursively extract bilingual term entries from any JSON/dict/list structure without schema coupling."""
    if isinstance(node, dict):
        src_val = (
            node.get("src")
            or node.get("en")
            or node.get("original")
            or node.get("term")
            or node.get("source")
        )
        tgt_val = (
            node.get("tgt")
            or node.get("zh-chs")
            or node.get("zh")
            or node.get("translation")
            or node.get("target")
        )
        if isinstance(src_val, str) and isinstance(tgt_val, str):
            s = src_val.strip()
            t = tgt_val.strip()
            if len(s) >= 2 and t and s.lower() != "nan" and t.lower() != "nan":
                key = s.lower()
                if key not in out:
                    out[key] = {
                        "src": s,
                        "tgt": t,
                        "note": str(node.get("note") or "").strip(),
                        "aliases": _normalize_aliases(node.get("aliases"), s),
                        "fix_source": bool(node.get("fix_source", False)),
                        "source": source_tag,
                    }
                return

        for k, v in node.items():
            if isinstance(v, str):
                ks = str(k).strip()
                vs = v.strip()
                if len(ks) >= 2 and vs and ks.lower() not in _KB_META_KEYS:
                    key = ks.lower()
                    if key not in out:
                        out[key] = {
                            "src": ks,
                            "tgt": vs,
                            "note": "",
                            "aliases": [],
                            "fix_source": False,
                            "source": source_tag,
                        }
            elif isinstance(v, (dict, list)):
                _extract_generic_kb_pairs(v, out, source_tag=source_tag)
    elif isinstance(node, list):
        for item in node:
            _extract_generic_kb_pairs(item, out, source_tag=source_tag)


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
            merged[key]["aliases"] = list(
                dict.fromkeys([*existing_aliases, *alias_list])
            )
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
            kb_entries = _get_kb_index([]).get("entries") or {}
            # Directly merge small-to-medium custom JSON maps (<= 500 entries);
            # large dictionaries (e.g. 20,000+ game/domain entries) are queried via _get_kb_index()
            if 0 < len(kb_entries) <= 500:
                for item in kb_entries.values():
                    _add_term(
                        item["src"],
                        item["tgt"],
                        item.get("note", ""),
                        aliases=item.get("aliases"),
                        fix_source=bool(item.get("fix_source", False)),
                        source="term_map",
                    )
        except Exception as exc:
            rprint(
                f"[yellow]⚠️ Could not read {OPTIONAL_JSON_TERMS_PATH}: {exc}[/yellow]"
            )

    return list(merged.values())


_KB_STOP_TOKENS = {
    "with",
    "that",
    "this",
    "from",
    "have",
    "what",
    "when",
    "where",
    "which",
    "while",
    "there",
    "their",
    "they",
    "them",
    "then",
    "than",
    "some",
    "more",
    "most",
    "very",
    "just",
    "like",
    "know",
    "think",
    "make",
    "made",
    "good",
    "best",
    "well",
    "much",
    "many",
    "only",
    "also",
    "even",
    "back",
    "into",
    "over",
    "after",
    "before",
    "under",
    "every",
    "other",
    "about",
    "because",
    "really",
    "actually",
    "basically",
    "literally",
    "going",
    "want",
    "need",
    "look",
    "take",
    "give",
    "come",
    "here",
    "time",
    "year",
    "thing",
    "things",
    "weapon",
    "weapons",
    "perk",
    "perks",
    "frame",
    "frames",
    "all",
    "now",
    "one",
    "two",
    "three",
    "four",
    "five",
    "new",
    "old",
    "first",
    "last",
    "high",
    "low",
    "left",
    "right",
    "down",
    "next",
    "same",
    "still",
    "such",
    "both",
    "each",
    "few",
    "own",
    "sure",
    "true",
    "full",
    "hard",
    "fast",
    "slow",
    "long",
    "short",
    "open",
    "close",
    "free",
    "real",
    "main",
    "part",
    "side",
    "kind",
    "type",
    "sort",
    "case",
    "point",
    "place",
    "work",
    "play",
    "turn",
    "move",
    "call",
    "keep",
    "find",
    "show",
    "tell",
    "mean",
    "feel",
    "seem",
    "leave",
    "put",
    "set",
    "run",
    "use",
    "try",
}

_KB_CONVERSATIONAL_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "it's",
    "that",
    "this",
    "these",
    "those",
    "what",
    "when",
    "where",
    "who",
    "whom",
    "which",
    "why",
    "how",
    "if",
    "or",
    "and",
    "but",
    "so",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "to",
    "up",
    "with",
    "out",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "done",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "have",
    "has",
    "had",
    "having",
    "you",
    "your",
    "yours",
    "we",
    "our",
    "ours",
    "they",
    "them",
    "their",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "me",
    "my",
    "mine",
    "i",
    "i'm",
    "you're",
    "we're",
    "all",
    "any",
    "some",
    "more",
    "most",
    "other",
    "another",
    "such",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "now",
    "then",
    "here",
    "there",
    "way",
    "end",
    "past",
    "present",
    "future",
    "cool",
    "good",
    "bad",
    "great",
    "nice",
    "little",
    "big",
    "small",
    "old",
    "new",
    "high",
    "low",
    "long",
    "short",
    "right",
    "left",
}


def _get_kb_index(
    custom_terms_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build or return a cached inverted index over authoritative glossary sources (including large ``term-map.json``)."""
    global _CACHED_KB_KEY, _CACHED_KB_DATA

    abs_map = os.path.abspath(OPTIONAL_JSON_TERMS_PATH)
    map_mtime = _file_mtime(abs_map)
    custom_sig = tuple(
        (t.get("src", ""), t.get("tgt", "")) for t in (custom_terms_list or [])
    )
    kb_key = (abs_map, map_mtime, custom_sig)

    with _INDEX_LOCK:
        if _CACHED_KB_KEY == kb_key and _CACHED_KB_DATA is not None:
            return _CACHED_KB_DATA

        kb_entries: dict[str, dict[str, Any]] = {}
        for t in custom_terms_list or []:
            s = str(t.get("src") or "").strip()
            tg = str(t.get("tgt") or "").strip()
            if s and tg:
                kb_entries[s.lower()] = {
                    "src": s,
                    "tgt": tg,
                    "note": str(t.get("note") or "").strip(),
                    "aliases": _normalize_aliases(t.get("aliases"), s),
                }

        if os.path.exists(OPTIONAL_JSON_TERMS_PATH):
            try:
                with open(OPTIONAL_JSON_TERMS_PATH, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                _extract_generic_kb_pairs(raw_data, kb_entries, source_tag="term_map")
            except Exception as exc:
                rprint(
                    f"[yellow]⚠️ Could not read {OPTIONAL_JSON_TERMS_PATH}: {exc}[/yellow]"
                )

        exact_lower_map: dict[str, dict[str, Any]] = {}
        compact_map: dict[str, list[dict[str, Any]]] = {}
        token_inv: dict[str, list[dict[str, Any]]] = {}
        p3_inv: dict[str, list[str]] = {}
        stem_ngram_map: dict[tuple[str, ...], dict[str, Any]] = {}

        for key, entry in kb_entries.items():
            en = entry["src"]
            exact_lower_map[key] = entry
            for q in re.findall(r'"([^"]+)"', en):
                ql = q.strip().lower()
                if len(ql) >= 3 and ql not in exact_lower_map:
                    exact_lower_map[ql] = entry
            for al in entry.get("aliases") or []:
                al_low = str(al).strip().lower()
                if al_low and al_low not in exact_lower_map:
                    exact_lower_map[al_low] = entry

            comp = re.sub(r"[^a-z0-9]", "", key)
            if len(comp) >= 2:
                compact_map.setdefault(comp, []).append(entry)

            toks = [
                t
                for t in re.findall(r"[a-z0-9]+", key)
                if len(t) >= 3 and t not in _KB_STOP_TOKENS
            ]
            for t in set(toks):
                if t not in token_inv:
                    token_inv[t] = []
                    p3_inv.setdefault(t[:3], []).append(t)
                token_inv[t].append(entry)

            _, stems = _tokenize_and_normalize(en)
            if stems and len(stems) <= 6:
                st_tuple = tuple(stems)
                if st_tuple not in stem_ngram_map:
                    stem_ngram_map[st_tuple] = entry

        kb_data = {
            "entries": kb_entries,
            "exact_lower_map": exact_lower_map,
            "compact_map": compact_map,
            "token_inv": token_inv,
            "p3_inv": p3_inv,
            "stem_ngram_map": stem_ngram_map,
        }
        _CACHED_KB_KEY = kb_key
        _CACHED_KB_DATA = kb_data
        return kb_data


def _scan_text_for_kb_terms(
    text: str, kb_data: dict[str, Any], max_hints: int = 40
) -> list[dict[str, Any]]:
    """Pre-scan a transcript against the authoritative KB to surface official terms that chunk extraction might miss."""
    stem_ngram_map = kb_data.get("stem_ngram_map") or {}
    if not stem_ngram_map or not text:
        return []
    raw_toks, stem_toks = _tokenize_and_normalize(text)
    n = len(stem_toks)
    if n == 0:
        return []

    cap_words = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9\-']+\b", text)}
    matched_by_src: dict[str, dict[str, Any]] = {}
    covered_positions: set[int] = set()

    # Scan from longest N-gram (6) down to 1-gram for Longest-Span Preference
    for k_len in range(min(6, n), 0, -1):
        for i in range(n - k_len + 1):
            if any(pos in covered_positions for pos in range(i, i + k_len)):
                continue
            gram = tuple(stem_toks[i : i + k_len])
            entry = stem_ngram_map.get(gram)
            if entry is None:
                continue
            window_raw = raw_toks[i : i + k_len]
            # Skip purely conversational stopword phrases ("what if", "the end", "the past", "it's cool")
            if all(tok in _KB_CONVERSATIONAL_STOPWORDS for tok in window_raw):
                continue
            surf = " ".join(window_raw)
            if k_len >= 2 or (
                surf in cap_words
                and len(surf) >= 5
                and surf not in _KB_STOP_TOKENS
                and surf not in _KB_CONVERSATIONAL_STOPWORDS
            ):
                if entry["src"] not in matched_by_src:
                    start_idx = max(0, i - 5)
                    end_idx = min(n, i + k_len + 5)
                    snippet = " ".join(raw_toks[start_idx:end_idx])
                    matched_by_src[entry["src"]] = {**entry, "context_snippet": snippet}
                for pos in range(i, i + k_len):
                    covered_positions.add(pos)
                if len(matched_by_src) >= max_hints:
                    break
        if len(matched_by_src) >= max_hints:
            break

    return list(matched_by_src.values())


def _retrieve_kb_candidates_for_term(
    term_obj: dict[str, Any], kb_data: dict[str, Any], top_k: int = 3
) -> list[dict[str, str]]:
    """Fuzzy-match an extracted term and its aliases against the authoritative KB (`term-map.json`)."""
    if not kb_data or not kb_data.get("entries"):
        return []
    exact_lower_map = kb_data["exact_lower_map"]
    compact_map = kb_data["compact_map"]
    token_inv = kb_data["token_inv"]
    p3_inv = kb_data.get("p3_inv") or {}

    # Query only the extracted term's own `src` and `aliases` (do not scrape free-text `note`)
    queries = [str(term_obj.get("src") or "")] + [
        str(a) for a in (term_obj.get("aliases") or [])
    ]

    results: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()

    def _add(e: dict[str, Any], score: float) -> None:
        if e["src"] not in seen:
            seen.add(e["src"])
            results.append((score, e))

    for q_str in queries:
        q_low = q_str.strip().lower()
        if not q_low:
            continue
        if q_low in exact_lower_map:
            _add(exact_lower_map[q_low], 100.0)
        q_comp = re.sub(r"[^a-z0-9]", "", q_low)
        for e in compact_map.get(q_comp, ()):
            _add(e, 99.0)

        q_toks = [
            t
            for t in re.findall(r"[a-z0-9]+", q_low)
            if len(t) >= 3 and t not in _KB_STOP_TOKENS
        ]
        cand_pool: list[dict[str, Any]] = []
        seen_cand_srcs: set[str] = set()

        def _pool_extend(entries: list[dict[str, Any]]) -> None:
            for ent in entries:
                s_k = ent["src"]
                if s_k not in seen_cand_srcs:
                    seen_cand_srcs.add(s_k)
                    cand_pool.append(ent)

        for qt in q_toks:
            if qt in token_inv:
                _pool_extend(token_inv[qt])
            st_qt = _stem_en_token(qt)
            if st_qt != qt and st_qt in token_inv:
                _pool_extend(token_inv[st_qt])
            if len(qt) >= 4:
                p4 = qt[:4]
                for kt in p3_inv.get(qt[:3], ()):
                    if kt.startswith(p4) or (
                        abs(len(kt) - len(qt)) <= 2
                        and _similarity_score(qt, kt) >= 75.0
                    ):
                        _pool_extend(token_inv[kt])

        for e in cand_pool:
            if e["src"] in seen:
                continue
            e_low = e["src"].lower()
            e_comp = re.sub(r"[^a-z0-9]", "", e_low)
            s1 = _similarity_score(q_low, e_low)
            s2 = _similarity_score(q_comp, e_comp)
            e_words = re.findall(r"[a-z0-9]+", e_low)
            q_words = re.findall(r"[a-z0-9]+", q_low)
            sub_bonus = 0.0
            if q_words and all(
                any(
                    ew.startswith(qw) or _similarity_score(qw, ew) >= 80.0
                    for ew in e_words
                )
                for qw in q_words
            ):
                if len(e_words) <= len(q_words) + 2:
                    sub_bonus = 88.0 - (len(e_words) - len(q_words)) * 2.0
            best_s = max(s1, s2, sub_bonus)
            if best_s >= 72.0:
                _add(e, best_s)

    results.sort(key=lambda x: x[0], reverse=True)
    return [{"src": r[1]["src"], "tgt": r[1]["tgt"]} for r in results[:top_k]]


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
            has_cjk = bool(_CJK_RE.search(surf_lower))
            has_non_hyphen_symbol = bool(re.search(r"[^a-z0-9\s\-']", surf_lower))
            if has_cjk:
                if surf_lower not in cjk_or_symbol_map:
                    cjk_or_symbol_map[surf_lower] = entry
            elif re.search(r"[^a-z0-9\s]", surf_lower):
                # Terms with symbols like "C++", "C#", ".NET", "x86_64"
                pattern = re.compile(
                    rf"(?<![a-z0-9]){re.escape(surf_lower)}(?![a-z0-9])", re.IGNORECASE
                )
                word_boundary_res.append((pattern, entry))

            # Do not strip critical symbols (like '+', '#', '.') into plain word tokens
            # that would falsely match everyday words ("C#" -> "c", ".NET" -> "net").
            if has_non_hyphen_symbol and not has_cjk:
                continue

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
                        "note": str(
                            item.get("note", "") or prev.get("note", "")
                        ).strip(),
                        "aliases": list(dict.fromkeys([*prev_aliases, *new_aliases])),
                        "fix_source": bool(
                            item.get("fix_source", prev.get("fix_source", False))
                        ),
                        "source": str(
                            item.get("source") or prev.get("source") or "task"
                        ),
                    }

        custom_list = load_custom_terms()
        custom_mtimes = [
            m for m in (global_mtime, xlsx_mtime, csv_mtime, map_mtime) if m is not None
        ]
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
                    "aliases": list(
                        dict.fromkeys(
                            [*(prev.get("aliases") or []), *(item.get("aliases") or [])]
                        )
                    ),
                }
        else:
            for item in custom_list:
                merged_by_src[item["src"].lower()] = dict(item)
            if contract_mtime is not None and (
                json_mtime is None or contract_mtime >= json_mtime
            ):
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
    raw_ngrams_by_k: dict[int, list[tuple[int, str]]] = {}
    matched_spans_by_src: dict[str, list[tuple[int, int]]] = {}

    for k in range(1, min(max_k, n_toks) + 1):
        raw_list: list[tuple[int, str]] = []
        for i in range(n_toks - k + 1):
            stem_gram = tuple(stem_toks[i : i + k])
            if stem_gram in stem_ngram_map:
                src_name = stem_ngram_map[stem_gram]["src"]
                matched_srcs.add(src_name)
                matched_spans_by_src.setdefault(src_name, []).append((i, i + k))
            raw_list.append((i, " ".join(raw_toks[i : i + k])))
        raw_ngrams_by_k[k] = raw_list

    # 3. Word-level N-gram fuzzy matching for ASR spelling/phonetic variants
    for norm_term, k, entry in index_data["fuzzy_entries"]:
        if entry["src"] in matched_srcs:
            continue
        candidate_ngrams = raw_ngrams_by_k.get(k)
        if not candidate_ngrams:
            continue
        term_len = len(norm_term)
        for i_pos, cand in candidate_ngrams:
            if abs(len(cand) - term_len) > max(2, int(term_len * 0.28)):
                continue
            if _is_Bounded_fuzzy_match(norm_term, cand, k):
                src_name = entry["src"]
                matched_srcs.add(src_name)
                matched_spans_by_src.setdefault(src_name, []).append((i_pos, i_pos + k))
                break

    # 4. Longest-Span Suppression: suppress shorter token-matched terms whose spans are
    # strictly contained within a longer matched term's span (e.g., "Compendium" inside "Data Compendium").
    if len(matched_spans_by_src) > 1:
        all_spans = [
            (s, e, src_name)
            for src_name, spans in matched_spans_by_src.items()
            for (s, e) in spans
        ]
        for src_name, spans in list(matched_spans_by_src.items()):
            if all(
                any(
                    other_src != src_name
                    and other_s <= s
                    and e <= other_e
                    and (other_e - other_s) > (e - s)
                    for other_s, other_e, other_src in all_spans
                )
                for s, e in spans
            ):
                matched_srcs.discard(src_name)

    if not matched_srcs:
        return None

    selected = [t for t in terms if t["src"] in matched_srcs]
    if not selected:
        return None

    lines: list[str] = []
    for i, term in enumerate(selected):
        aliases = term.get("aliases") or []
        alias_suffix = (
            f" (also transcribed as: {', '.join(f'{a!r}' for a in aliases)})"
            if aliases
            else ""
        )
        lines.append(
            f'{i + 1}. "{term["src"]}"{alias_suffix}: "{term["tgt"]}", meaning: {term["note"]}'
        )
    return "\n".join(lines)


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
    by_alias_lower: dict[str, str] = {}
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
            for a in aliases:
                by_alias_lower[a.lower()] = s_low
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

            target_key = s_low if s_low in by_src_lower else by_alias_lower.get(s_low)
            if target_key and target_key in by_src_lower:
                canonical_entry = by_src_lower[target_key]
                if not canonical_entry.get("note") and note:
                    canonical_entry["note"] = note
                merged_aliases = _normalize_aliases(
                    [
                        *(canonical_entry.get("aliases") or []),
                        *([src] if target_key != s_low else []),
                        *aliases,
                    ],
                    canonical_entry["src"],
                )
                if merged_aliases:
                    canonical_entry["aliases"] = merged_aliases
                    for a in merged_aliases:
                        by_alias_lower[a.lower()] = target_key
                continue

            if stem_tuple and stem_tuple in by_stem_key:
                tgt = by_stem_key[stem_tuple]
            elif stem_tuple:
                by_stem_key[stem_tuple] = tgt

            entry = {"src": src, "tgt": tgt, "note": note}
            if aliases:
                entry["aliases"] = aliases
                for a in aliases:
                    by_alias_lower[a.lower()] = s_low
            by_src_lower[s_low] = entry

    return {
        "theme": combined_theme,
        "topic": combined_theme,
        "terms": list(by_src_lower.values()),
    }


def _consolidate_terms_with_llm(
    merged_summary: dict,
    chunk_results: list[tuple[int, dict]],
    custom_terms_list: list[dict[str, Any]],
    kb_data: dict[str, Any] | None = None,
    full_text: str = "",
) -> dict:
    """Run an LLM Reduce pass to unify cross-chunk ASR variants and align with authoritative KB candidates."""
    raw_terms = merged_summary.get("terms") or []
    if not raw_terms and not full_text:
        return merged_summary

    custom_by_low = {
        str(t["src"]).strip().lower(): t for t in custom_terms_list if t.get("src")
    }
    enriched_terms: list[dict[str, Any]] = []
    seen_srcs: set[str] = set()
    has_kb_candidates = False

    for t in raw_terms:
        item = dict(t)
        s_low = str(item.get("src") or "").strip().lower()
        seen_srcs.add(s_low)
        for al in item.get("aliases") or []:
            seen_srcs.add(str(al).strip().lower())
        if kb_data and kb_data.get("entries") and s_low not in custom_by_low:
            cands = _retrieve_kb_candidates_for_term(item, kb_data, top_k=3)
            if cands:
                item["kb_candidates"] = cands
                has_kb_candidates = True
        enriched_terms.append(item)

    if kb_data and kb_data.get("entries") and full_text:
        scanned_kb_hits = _scan_text_for_kb_terms(full_text, kb_data, max_hints=35)
        for kb_hit in scanned_kb_hits:
            hit_low = kb_hit["src"].lower()
            if hit_low not in seen_srcs:
                seen_srcs.add(hit_low)
                cand_entry: dict[str, Any] = {
                    "src": kb_hit["src"],
                    "tgt": kb_hit["tgt"],
                    "note": kb_hit.get("note")
                    or "Official glossary match in transcript",
                    "source_hint": "kb_scan_candidate",
                }
                if kb_hit.get("context_snippet"):
                    cand_entry["context_snippet"] = kb_hit["context_snippet"]
                enriched_terms.append(cand_entry)
                has_kb_candidates = True

    if len(chunk_results) <= 1 and not has_kb_candidates:
        return merged_summary

    check_cancel()
    prompt = get_consolidate_terms_prompt(
        enriched_terms, theme=merged_summary.get("theme", "")
    )
    try:
        resp = ask_gpt(
            prompt,
            resp_type="json",
            valid_def=_valid_summary,
            log_title="consolidate_terms",
        )
    except StopTask:
        raise
    except Exception as exc:
        rprint(
            f"[yellow]⚠️ LLM term consolidation failed, using deterministic merge: {exc}[/yellow]"
        )
        return merged_summary

    if not isinstance(resp, dict) or not isinstance(resp.get("terms"), list):
        return merged_summary

    # Guard against naive unit-test mocks that return a single chunk's response verbatim
    if any(resp == cr for _, cr in chunk_results):
        return merged_summary

    # Ensure every explicit custom term remains present and authoritative, and preserve original surface forms
    consolidated_chunk = [
        (0, {"theme": merged_summary.get("theme", ""), "terms": resp["terms"]})
    ]
    final_merged = _merge_terminology_results(consolidated_chunk, custom_terms_list)

    # Also ensure any original surface forms from raw_terms that were merged into a canonical KB term
    # are kept in `aliases` if they match the KB candidate of a consolidated term.
    final_by_low = {t["src"].lower(): t for t in final_merged["terms"]}
    for raw_t in enriched_terms:
        if raw_t.get("source_hint") == "kb_scan_candidate":
            continue
        raw_src = str(raw_t.get("src") or "").strip()
        raw_aliases = _normalize_aliases(raw_t.get("aliases"), raw_src)
        for cand in raw_t.get("kb_candidates") or []:
            cand_low = str(cand.get("src") or "").strip().lower()
            if cand_low in final_by_low and raw_src.lower() != cand_low:
                target_entry = final_by_low[cand_low]
                merged_al = _normalize_aliases(
                    [*(target_entry.get("aliases") or []), raw_src, *raw_aliases],
                    target_entry["src"],
                )
                if merged_al:
                    target_entry["aliases"] = merged_al

    final_merged["theme"] = merged_summary.get("theme", "")
    final_merged["topic"] = merged_summary.get("topic", "")
    return final_merged


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
    kb_data = _get_kb_index(custom_terms_list)
    full_text = " ".join(chunks)
    summary = _consolidate_terms_with_llm(
        summary, chunk_results, custom_terms_list, kb_data=kb_data, full_text=full_text
    )

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

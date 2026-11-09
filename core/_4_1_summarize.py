from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any
import unicodedata

from nltk.stem import SnowballStemmer
import pandas as pd

from core.concurrency import ContextThreadPoolExecutor
from core.prompts import get_kb_disambiguation_prompt, get_summary_prompt
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
_UNICODE_SEP_RE = re.compile(r"[\u2010-\u2015\u2212\u2044/]+")
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

_EXTRA_COMMON_WORDS: frozenset[str] = frozenset(
    (
        "ll ve re dont doesnt didnt cant wont isnt arent wasnt werent "
        "hasnt havent hadnt wouldnt couldnt shouldnt mightnt mustnt "
        "gonna wanna gotta kinda sorta yall im ive id youll youve youd "
        "hell hes hed shell shes shed well weve wed theyll theyve theyd "
        "thats whats whos wheres whens whys hows lets "
        "dps pve pvp dot aoe ads rng hud ui ux "
        "reprisal impetus doubt call end story clash control rumble "
        "supremacy mayhem breakthrough countdown survival elimination "
        "salvage momentum rift doubles inevitability inevitably"
    ).split()
)


def _fold_ascii(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", _UNICODE_SEP_RE.sub(" ", text))
        .encode("ascii", "ignore")
        .decode("ascii")
    )


@lru_cache(maxsize=65536)
def _stem_word(word: str) -> str:
    w = word.lower()
    if len(w) <= 2 or w.endswith(("ism", "ancy", "ency")):
        return w
    st = _STEMMER.stem(w)
    if st.endswith("lv") and len(st) >= 4:
        st = st[:-1] + "f"
    elif len(st) >= 3 and st[-1] == st[-2] and st[-1] in "flsz":
        st = st[:-1]
    return st


@lru_cache(maxsize=1)
def _load_common_english_words() -> frozenset[str]:
    words: set[str] = set(_EXTRA_COMMON_WORDS)
    for dict_path in ("/usr/share/dict/words", "/usr/dict/words"):
        p = Path(dict_path)
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    w = line.strip()
                    if len(w) >= 3 and w.islower() and w.isalpha():
                        words.add(w)
            except OSError:
                pass
            break
    return frozenset(words)


@lru_cache(maxsize=65536)
def _is_standard_english(word: str) -> bool:
    w = word.lower()
    if len(w) <= 2 or w in _KB_STOP_TOKENS:
        return True
    common = _load_common_english_words()
    if w in common or _stem_word(w) in common:
        return True
    for suf in (
        "s",
        "es",
        "ed",
        "ing",
        "er",
        "ers",
        "est",
        "ly",
        "ness",
        "ment",
        "able",
        "ible",
    ):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if base in common or (base + "e") in common:
                return True
            if len(base) >= 4 and base[-1] == base[-2] and base[:-1] in common:
                return True
            if base.endswith("i") and (base[:-1] + "y") in common:
                return True
    return False


def _levenshtein_fast(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        cur = [i]
        for j, c2 in enumerate(s2, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (c1 != c2)
            cur.append(
                ins if ins < dele and ins < sub else (dele if dele < sub else sub)
            )
        prev = cur
    return prev[-1]


@lru_cache(maxsize=65536)
def _phonetic_skeleton(word: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", word.lower())
    if not s:
        return ""
    s = re.sub(r"^[aeiouy]+", "a", s)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = (
        s.replace("ph", "f")
        .replace("gh", "g")
        .replace("kn", "n")
        .replace("wr", "r")
        .replace("wh", "w")
        .replace("ck", "k")
        .replace("dg", "j")
        .replace("tch", "ch")
        .replace("sh", "s")
        .replace("th", "t")
        .replace("qu", "kw")
    )
    s = (
        s.replace("c", "k")
        .replace("q", "k")
        .replace("x", "ks")
        .replace("z", "s")
        .replace("v", "f")
    )
    first = s[0]
    rest = re.sub(r"[aeiouyhw]", "", s[1:])
    raw = first + rest
    dedup: list[str] = []
    for ch in raw:
        if not dedup or dedup[-1] != ch:
            dedup.append(ch)
    return "".join(dedup)


def _char_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 1.0 - _levenshtein_fast(a, b) / max(len(a), len(b))


def _word_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b or _stem_word(a) == _stem_word(b):
        return 1.0
    c_sim = _char_sim(a, b)
    pa, pb = _phonetic_skeleton(a), _phonetic_skeleton(b)
    p_sim = _char_sim(pa, pb) if (len(pa) >= 2 and len(pb) >= 2) else 0.0
    if (
        len(pa) >= 3
        and len(pb) >= 3
        and pa[0] == pb[0]
        and not _is_standard_english(a)
        and not _is_standard_english(b)
    ):
        da = pa[0] + re.sub(r"(.)\1+", r"\1", pa[1:].replace("d", "t"))
        db = pb[0] + re.sub(r"(.)\1+", r"\1", pb[1:].replace("d", "t"))
        p_sim = max(p_sim, _char_sim(da, db) * 0.96)
    p_weighted = (
        p_sim * 0.94
        if (len(pa) >= 2 and len(pb) >= 2 and pa[0] == pb[0])
        else p_sim * 0.80
    )
    return max(c_sim, p_weighted, 0.55 * c_sim + 0.45 * p_sim)


def _tokenize_surface(text: str) -> tuple[list[str], list[str], list[str]]:
    folded = _fold_ascii(text).lower()
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


def _span_sim(
    query: str,
    target_bw: tuple[str, ...],
    target_st: tuple[str, ...],
    target_join: str,
    stem_df: dict[str, int],
) -> float:
    q_bw, q_st, _ = _tokenize_surface(query)
    if not q_bw or not target_bw:
        return 0.0
    if tuple(q_bw) == target_bw or tuple(q_st) == target_st:
        return 1.0
    q_join = "".join(q_bw)
    j_sim = _word_sim(q_join, target_join)
    if len(q_bw) == 1 and len(target_bw) >= 2:
        t_sims = [
            _word_sim(q_bw[0], tw)
            for tw, ts in zip(target_bw, target_st)
            if len(tw) >= 5
            and ts not in _KB_STOP_TOKENS
            and not _is_standard_english(tw)
            and stem_df.get(ts, 99) <= 3
        ]
        best_tok = max(t_sims) if t_sims else 0.0
        return max(j_sim, best_tok * 0.94)
    if len(q_bw) != len(target_bw):
        return j_sim * 0.92
    tok_sims = [_word_sim(qw, tw) for qw, tw in zip(q_bw, target_bw)]
    if min(tok_sims) < 0.55:
        return j_sim * 0.85
    avg_tok = sum(tok_sims) / len(tok_sims)
    return max(avg_tok, j_sim * 0.95)


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
            t = re.sub(r"\s*\n\s*", "", tgt_val).strip()
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
                t = re.sub(r"\s*\n\s*", "", v).strip()
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


def _normalize_kb_entries(
    kb_entries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, entry in kb_entries.items():
        clean_tgt = re.sub(r"\s*\n\s*", "", str(entry.get("tgt") or "")).strip()
        als = list(entry.get("aliases") or [])
        src = str(entry.get("src") or "").strip()
        if "-" in src:
            for part in src.split():
                if (
                    "-" in part[1:-1]
                    and part[0].isalnum()
                    and len(part) >= 6
                    and part.lower() != src.lower()
                ):
                    if part.lower() not in {a.lower() for a in als}:
                        als.append(part)
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
            "aliases": _normalize_aliases(als, src),
        }

    for k, entry in list(out.items()):
        src = entry["src"]
        tgt = entry["tgt"]
        if ":" in src and ("：" in tgt or ":" in tgt):
            s_parts = [p.strip() for p in src.split(":") if p.strip()]
            t_parts = [p.strip() for p in re.split(r"[：:]", tgt) if p.strip()]
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


def _char_phonetic_ngrams(text: str) -> list[str]:
    bw, _, _ = _tokenize_surface(text)
    if not bw:
        return []
    joined = "".join(bw)
    skel = "".join(_phonetic_skeleton(w) for w in bw)
    grams: list[str] = []
    padded = f"^{joined}$"
    for n in (2, 3):
        for i in range(len(padded) - n + 1):
            grams.append(f"c:{padded[i : i + n]}")
    if skel:
        ps = f"^{skel}$"
        for n in range(2, min(4, len(ps) + 1)):
            for i in range(len(ps) - n + 1):
                grams.append(f"p:{ps[i : i + n]}")
        for w in bw:
            pw = _phonetic_skeleton(w)
            if len(pw) >= 2:
                grams.append(f"pw:{pw}")
                grams.append(f"p2:{pw[:2]}")
    return grams


class UnifiedNgramKBIndex:
    """Indexes canonical surfaces, curated aliases, and unique proper-noun head tokens (df <= 3)."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.surfaces: list[tuple[str, dict[str, Any], float]] = []
        tok_df: dict[str, int] = {}
        tokenized_entries: list[tuple[dict[str, Any], list[str]]] = []

        for entry in entries:
            src = str(entry.get("src") or "").strip()
            if not src:
                continue
            bw, _, _ = _tokenize_surface(src)
            tokenized_entries.append((entry, bw))
            for w in set(bw):
                if len(w) >= 4 and w not in _KB_STOP_TOKENS:
                    tok_df[w] = tok_df.get(w, 0) + 1

        for entry, bw in tokenized_entries:
            src = entry["src"]
            penalty = 0.92 if ("(" in src or ":" in src) else 1.0
            self.surfaces.append((src, entry, penalty))
            for al in entry.get("aliases") or []:
                al_s = str(al).strip()
                if al_s:
                    self.surfaces.append((al_s, entry, penalty * 0.98))
            if len(bw) >= 2 and "(" not in src and ":" not in src:
                for w in bw:
                    if (
                        len(w) >= 5
                        and w not in _KB_STOP_TOKENS
                        and tok_df.get(w, 99) <= 3
                        and not _is_standard_english(w)
                    ):
                        self.surfaces.append((w, entry, 0.94))

        df: dict[str, int] = {}
        raw_grams_per_surf: list[list[str]] = []
        for surf_text, _, _ in self.surfaces:
            g = _char_phonetic_ngrams(surf_text)
            raw_grams_per_surf.append(g)
            for tok in set(g):
                df[tok] = df.get(tok, 0) + 1

        n_docs = max(1, len(self.surfaces))
        self.idf = {
            tok: math.log(1.0 + n_docs / (1.0 + freq)) for tok, freq in df.items()
        }
        self.surf_norms: list[float] = []
        self.inv: dict[str, list[tuple[int, float]]] = {}

        for surf_id, g_list in enumerate(raw_grams_per_surf):
            if not g_list:
                self.surf_norms.append(0.0)
                continue
            tf: dict[str, int] = {}
            for tok in g_list:
                tf[tok] = tf.get(tok, 0) + 1
            vec = {
                tok: (1.0 + math.log(cnt)) * self.idf[tok] for tok, cnt in tf.items()
            }
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self.surf_norms.append(norm)
            for tok, w in vec.items():
                self.inv.setdefault(tok, []).append((surf_id, w))

    def query(
        self, span: str, top_k: int = 3, min_score: float = 0.45
    ) -> list[tuple[float, dict[str, Any]]]:
        g_list = _char_phonetic_ngrams(span)
        if not g_list:
            return []
        tf: dict[str, int] = {}
        for tok in g_list:
            tf[tok] = tf.get(tok, 0) + 1
        q_vec = {
            tok: (1.0 + math.log(cnt)) * self.idf.get(tok, 1.0)
            for tok, cnt in tf.items()
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0:
            return []

        dots: dict[int, float] = {}
        for tok, qw in q_vec.items():
            for surf_id, dw in self.inv.get(tok, ()):
                dots[surf_id] = dots.get(surf_id, 0.0) + qw * dw

        best_by_src: dict[str, tuple[float, dict[str, Any]]] = {}
        for surf_id, dot in dots.items():
            d_norm = self.surf_norms[surf_id]
            if d_norm <= 0.0:
                continue
            _, entry, penalty = self.surfaces[surf_id]
            sc = (dot / (q_norm * d_norm)) * penalty
            if sc < min_score:
                continue
            sl = entry["src"].lower()
            prev = best_by_src.get(sl)
            if prev is None or sc > prev[0]:
                best_by_src[sl] = (sc, entry)

        scored = sorted(best_by_src.values(), key=lambda x: x[0], reverse=True)
        return scored[:top_k]


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
    stem_df: dict[str, int] = {}
    src_records: list[tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str]] = []

    for entry in entries:
        src = str(entry.get("src") or "").strip()
        tgt = str(entry.get("tgt") or "").strip()
        if not src or not tgt:
            continue
        sig = (src.lower(), tgt.lower())
        aliases = _normalize_aliases(entry.get("aliases"), src)
        surface_specs = [(src, True), *((al, False) for al in aliases)]
        seen_st_for_df: set[str] = set()

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

            if is_src or len(joined_bw) >= 4:
                src_records.append((entry, tuple(bw), tuple(st), joined_bw))
                for tok in set(st):
                    if (
                        len(tok) >= 3
                        and tok not in _KB_STOP_TOKENS
                        and tok not in seen_st_for_df
                    ):
                        seen_st_for_df.add(tok)
                        stem_df[tok] = stem_df.get(tok, 0) + 1
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

    anchor_inv: dict[
        str, list[tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str]]
    ] = {}
    for entry, bw_tup, st_tup, joined_bw in src_records:
        if len(bw_tup) < 2:
            continue
        content_stems = [
            s for s in set(st_tup) if len(s) >= 3 and s not in _KB_STOP_TOKENS
        ]
        if len(content_stems) < 2:
            continue
        for s in content_stems:
            if stem_df.get(s, 0) <= 25:
                anchor_inv.setdefault(s, []).append((entry, bw_tup, st_tup, joined_bw))

    rare_by_p0: dict[str, list[tuple[str, tuple[dict[str, Any], ...]]]] = {}
    for rt, entries_tup in rare_token_inv.items():
        if len(rt) >= 6 and not _is_standard_english(rt):
            p_rt = _phonetic_skeleton(rt)
            if p_rt:
                rare_by_p0.setdefault(p_rt[0], []).append((rt, entries_tup))

    return {
        "exact_ngram_map": {k: tuple(v) for k, v in exact_ngram_map.items()},
        "stem_ngram_map": {k: tuple(v) for k, v in stem_ngram_map.items()},
        "first_stem_lengths": {k: frozenset(v) for k, v in first_stem_lengths.items()},
        "compact_stem_map": {k: tuple(v) for k, v in compact_stem_map.items()},
        "symbol_cjk_terms": tuple(symbol_cjk_terms),
        "rare_token_inv": rare_token_inv,
        "stem_df": stem_df,
        "anchor_inv": {k: tuple(v) for k, v in anchor_inv.items()},
        "rare_by_p0": {k: tuple(v) for k, v in rare_by_p0.items()},
        "surface_records": tuple(src_records),
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

    folded = _fold_ascii(text)
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
        note = str(hit.entry.get("note") or "").strip()
        note_suffix = f" [{note}]" if note else ""
        src_lines = [f"{src}:{tgt}{note_suffix}"] if hit.src_matched else []
        alias_lines: list[str] = []
        for al in hit.matched_aliases:
            al_note = note_suffix
            if (
                al.lower().endswith("less")
                and not src.lower().endswith("less")
                and "without" not in note.lower()
            ):
                al_note = f" [without {src}; {note}]" if note else f" [without {src}]"
            alias_lines.append(f"{al}:{tgt} ({src}){al_note}")
        for line in (*src_lines, *alias_lines):
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

        kb_entries = _normalize_kb_entries(kb_entries)
        entries_list = list(kb_entries.values())
        matcher = _build_term_matcher(entries_list)
        ngram_index = UnifiedNgramKBIndex(entries_list)
        data = {
            "entries": kb_entries,
            "matcher": matcher,
            "ngram_index": ngram_index,
        }
        _KB_INDEX_CACHE["key"] = cache_key
        _KB_INDEX_CACHE["data"] = data
        return data


def _ensure_kb_matcher(kb_data: dict[str, Any]) -> dict[str, Any]:
    matcher = kb_data.get("matcher")
    if matcher is None or "surface_records" not in matcher:
        entries = list((kb_data.get("entries") or {}).values())
        matcher = _build_term_matcher(entries)
        kb_data["matcher"] = matcher
    if "ngram_index" not in kb_data and kb_data.get("entries"):
        kb_data["ngram_index"] = UnifiedNgramKBIndex(
            list((kb_data.get("entries") or {}).values())
        )
    return matcher


def _retrieve_rare_token_candidates(
    text: str,
    kb_data: dict[str, Any],
    exclude_srcs: set[str] | None = None,
    covered_tokens: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return []
    matcher = _ensure_kb_matcher(kb_data)
    ngram_idx: UnifiedNgramKBIndex | None = kb_data.get("ngram_index")
    folded = _fold_ascii(text)
    orig_words = _WORD_RE.findall(folded)
    if not orig_words:
        return []

    base_words: list[str] = []
    for w in orig_words:
        wl = w.lower()
        bw = re.sub(r"['’]s$", "", wl).replace("'", "").replace("’", "")
        base_words.append(bw)
    stems = [_stem_word(bw) if bw else "" for bw in base_words]
    n = len(base_words)

    if covered_tokens is None:
        covered_tokens = set()
        for hit in _match_terms_in_text(text, matcher):
            for surf in (hit.entry["src"], *hit.matched_aliases):
                for w in _tokenize_surface(surf)[0]:
                    covered_tokens.add(w)

    candidates_by_src: dict[str, dict[str, Any]] = {}
    aligned_oov_tokens: dict[str, dict[str, Any]] = {}

    def _add_cand(entry: dict[str, Any], alias_span: str) -> None:
        sl = entry["src"].lower()
        clean_al = alias_span.strip()
        if not clean_al or clean_al.lower() == sl:
            return
        item = candidates_by_src.get(sl)
        if item is None:
            als = list(entry.get("aliases") or [])
            if clean_al.lower() not in {a.lower() for a in als}:
                als.append(clean_al)
            candidates_by_src[sl] = {
                "src": entry["src"],
                "tgt": entry["tgt"],
                "note": str(entry.get("note") or ""),
                "aliases": als,
                "matched_spans": [clean_al],
                "fix_source": bool(entry.get("fix_source", False)),
                "source": str(entry.get("source") or "domain_kb"),
            }
        else:
            if clean_al.lower() not in {a.lower() for a in item["aliases"]}:
                item["aliases"].append(clean_al)
            if clean_al.lower() not in {a.lower() for a in item["matched_spans"]}:
                item["matched_spans"].append(clean_al)

    anchor_inv = matcher.get("anchor_inv", {})
    stem_df = matcher.get("stem_df", {})
    for i, st in enumerate(stems):
        if st not in anchor_inv:
            continue
        for entry, e_bw, e_st, _ in anchor_inv[st]:
            m = len(e_bw)
            for pos, tok_st in enumerate(e_st):
                if tok_st != st:
                    continue
                start = i - pos
                end = start + m
                if start < 0 or end > n:
                    continue
                win_bw = base_words[start:end]
                win_st = stems[start:end]
                if tuple(win_bw) == e_bw or tuple(win_st) == e_st:
                    continue
                tok_sims: list[float] = []
                ok = True
                for qw, qs, tw, ts in zip(win_bw, win_st, e_bw, e_st):
                    if qw == tw or qs == ts:
                        tok_sims.append(1.0)
                    elif ts in _KB_STOP_TOKENS and qw in _KB_STOP_TOKENS:
                        tok_sims.append(0.85)
                    else:
                        sim = _word_sim(qw, tw)
                        pq, pt = _phonetic_skeleton(qw), _phonetic_skeleton(tw)
                        both_std = _is_standard_english(qw) and _is_standard_english(tw)
                        if both_std:
                            if not pq or not pt or pq[0] != pt[0]:
                                ok = False
                                break
                            min_tok = 0.56 if stem_df.get(st, 99) <= 5 else 0.78
                            if sim < min_tok:
                                ok = False
                                break
                        elif sim < 0.56:
                            ok = False
                            break
                        tok_sims.append(sim)
                if ok and (sum(tok_sims) / m) >= 0.78:
                    span_str = " ".join(orig_words[start:end])
                    _add_cand(entry, span_str)
                    for qw, tw in zip(win_bw, e_bw):
                        if (
                            qw != tw
                            and len(qw) >= 5
                            and not _is_standard_english(qw)
                            and not _is_standard_english(tw)
                        ):
                            aligned_oov_tokens[qw] = entry

    rare_by_p0 = matcher.get("rare_by_p0", {})
    for i in range(n - 1):
        w1, w2 = base_words[i], base_words[i + 1]
        if len(w1) < 3 or len(w2) < 3 or w1 in _KB_STOP_TOKENS or w2 in _KB_STOP_TOKENS:
            continue
        if w1 in covered_tokens and w2 in covered_tokens:
            continue
        joined = w1 + w2
        if len(joined) < 6 or _is_standard_english(joined):
            continue
        p_join = _phonetic_skeleton(joined)
        if len(p_join) < 3:
            continue
        for rt, entries_list in rare_by_p0.get(p_join[0], ()):
            if abs(len(joined) - len(rt)) > 2:
                continue
            c_sim = _char_sim(joined, rt)
            w_sim = _word_sim(joined, rt)
            p_rt = _phonetic_skeleton(rt)
            min_c = 0.65 if (p_join == p_rt and len(p_join) >= 4) else 0.72
            if c_sim >= min_c and w_sim >= 0.85:
                span2 = " ".join(orig_words[i : i + 2])
                for entry in entries_list:
                    _add_cand(entry, span2)
                    _, e_st, _ = _tokenize_surface(entry["src"])
                    if i + 2 < n and stems[i + 2] in e_st:
                        _add_cand(entry, " ".join(orig_words[i : i + 3]))

    stem_ngram_map = matcher.get("stem_ngram_map", {})
    if ngram_idx is not None:
        for i in range(n - 1):
            w1, w2 = base_words[i], base_words[i + 1]
            if w1 in _KB_STOP_TOKENS or w2 in _KB_STOP_TOKENS:
                continue
            if w1 in covered_tokens and w2 in covered_tokens:
                continue
            if (stems[i], stems[i + 1]) in stem_ngram_map:
                continue
            if not (orig_words[i][0].isupper() and orig_words[i + 1][0].isupper()):
                continue
            span2 = f"{orig_words[i]} {orig_words[i + 1]}"
            for _, entry in ngram_idx.query(span2, top_k=1, min_score=0.47):
                e_bw, _, _ = _tokenize_surface(entry["src"])
                if len(e_bw) < 2:
                    continue
                if not all(
                    max(_word_sim(qw, tw) for tw in e_bw) >= 0.55 for qw in (w1, w2)
                ):
                    continue
                if len(e_bw) == 2:
                    mismatch_incompatible = False
                    for qw, tw in zip((w1, w2), e_bw):
                        if qw != tw and (
                            qw in covered_tokens or _word_sim(qw, tw) < 0.62
                        ):
                            mismatch_incompatible = True
                            break
                    if mismatch_incompatible:
                        continue
                _add_cand(entry, span2)

    rare_inv = matcher.get("rare_token_inv", {})
    surface_records = matcher.get("surface_records", ())
    seen_oov: set[str] = set()
    for i, bw in enumerate(base_words):
        if len(bw) < 4 or bw in _KB_STOP_TOKENS or bw.isdigit():
            continue
        if bw in covered_tokens or bw in seen_oov:
            continue
        is_oov = not _is_standard_english(bw)
        is_cap_noun = (
            orig_words[i][0].isupper()
            and len(bw) >= 7
            and not bw.endswith(("est", "ing", "ed", "ly", "tion", "ment"))
        )
        if not (is_oov or is_cap_noun):
            continue
        seen_oov.add(bw)
        if bw in aligned_oov_tokens:
            _add_cand(aligned_oov_tokens[bw], orig_words[i])
            continue
        st = stems[i]
        if is_oov and st in rare_inv:
            for entry in rare_inv[st]:
                _add_cand(entry, orig_words[i])
            continue
        scored_by_src: dict[str, tuple[float, dict[str, Any]]] = {}
        for entry, e_bw, e_st, e_join in surface_records:
            if (
                not is_oov
                and len(e_bw) == 1
                and _is_standard_english(entry["src"].lower())
            ):
                continue
            sc = _span_sim(bw, e_bw, e_st, e_join, stem_df)
            if sc >= 0.75:
                sl = entry["src"].lower()
                if "(" in sl:
                    sc *= 0.94
                prev = scored_by_src.get(sl)
                if prev is None or sc > prev[0]:
                    scored_by_src[sl] = (sc, entry)
        scored = sorted(scored_by_src.values(), key=lambda x: x[0], reverse=True)
        chosen_entries = [entry for _, entry in scored[:2]]
        if ngram_idx is not None:
            min_ng = 0.39 if is_oov else 0.54
            for _, entry in ngram_idx.query(orig_words[i], top_k=1, min_score=min_ng):
                if entry["src"].lower() not in {
                    e["src"].lower() for e in chosen_entries
                }:
                    chosen_entries.append(entry)
        for entry in chosen_entries:
            _add_cand(entry, orig_words[i])
            if i + 1 < n:
                _, e_st, _ = _tokenize_surface(entry["src"])
                if stems[i + 1] in e_st:
                    _add_cand(entry, " ".join(orig_words[i : i + 2]))

    entries_map = kb_data.get("entries", {})
    for sl, cand in list(candidates_by_src.items()):
        if " " not in sl and not _is_standard_english(sl):
            prefix = f"{sl}'s "
            for ek, e_entry in entries_map.items():
                if ek.startswith(prefix) and "(" not in ek:
                    _add_cand(e_entry, cand["src"])
                    for sp in cand.get("matched_spans", []):
                        _add_cand(e_entry, sp)

    return list(candidates_by_src.values())


def _scan_text_for_kb_terms(text: str, kb_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return []
    matcher = _ensure_kb_matcher(kb_data)
    hits = _match_terms_in_text(text, matcher)
    results: list[dict[str, Any]] = []
    for hit in hits:
        src = hit.entry["src"]
        source = str(hit.entry.get("source") or "domain_kb")
        if source == "domain_dict":
            bw, _, _ = _tokenize_surface(src)
            content_bw = [w for w in bw if w not in _KB_STOP_TOKENS]
            if len(content_bw) <= 1 and all(
                w in _EXTRA_COMMON_WORDS for w in content_bw
            ):
                continue
        results.append(
            {
                "src": src,
                "tgt": hit.entry["tgt"],
                "note": str(hit.entry.get("note") or ""),
                "aliases": list(hit.matched_aliases),
                "src_matched": hit.src_matched,
                "fix_source": bool(hit.entry.get("fix_source", False)),
                "source": source,
            }
        )
    return results


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


def _resolve_ambiguous_kb_candidates_with_llm(
    unresolved_items: list[dict[str, Any]],
) -> dict[str, str]:
    if not unresolved_items:
        return {}
    decisions: dict[str, str] = {}
    batch_size = 30
    batches = [
        unresolved_items[i : i + batch_size]
        for i in range(0, len(unresolved_items), batch_size)
    ]

    def _run_batch(batch: list[dict[str, Any]]) -> dict[str, str]:
        check_cancel()
        prompt = get_kb_disambiguation_prompt({"items": batch})
        resp = ask_gpt(prompt, resp_type="json", log_title="kb_disambiguate")
        out: dict[str, str] = {}
        if isinstance(resp, dict):
            if isinstance(resp.get("decisions"), list):
                for d in resp["decisions"]:
                    if isinstance(d, dict):
                        sp = str(d.get("span") or "").strip().lower()
                        ch = str(d.get("canonical_src") or "NONE").strip()
                        if sp and ch and ch.upper() != "NONE":
                            out[sp] = ch
            else:
                for idx, item in enumerate(batch, 1):
                    ch = str(resp.get(str(idx)) or "NONE").strip()
                    sp = str(item.get("span") or "").strip().lower()
                    if sp and ch and ch.upper() != "NONE":
                        out[sp] = ch
        return out

    max_workers = max(1, min(len(batches), int(load_key_or("max_workers", 4) or 4)))
    with ContextThreadPoolExecutor(max_workers=max_workers) as ex:
        for batch_out in ex.map(_run_batch, batches):
            decisions.update(batch_out)
    return decisions


def _merge_terminology_results(
    chunk_results: list[tuple[int, dict[str, Any]]],
    custom_terms_list: list[dict[str, Any]],
    kb_data: dict[str, Any] | None = None,
    full_text: str = "",
    run_llm_disambiguation: bool = False,
) -> dict[str, Any]:
    ordered_chunks = sorted(chunk_results, key=lambda x: x[0])
    themes: list[str] = []
    for _, res in ordered_chunks:
        if isinstance(res, dict):
            th = str(res.get("theme") or res.get("topic") or "").strip()
            if th and th not in themes:
                themes.append(th)
    combined_theme = " ".join(themes).strip()

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

    kb_entries = (kb_data.get("entries") or {}) if kb_data else {}
    if kb_entries:
        for k, entry in kb_entries.items():
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

    llm_mentioned_surfaces: set[str] = set()
    for _, res in ordered_chunks:
        if not isinstance(res, dict):
            continue
        for item in res.get("terms") or []:
            if not isinstance(item, dict):
                continue
            s = str(item.get("src") or "").strip()
            if s:
                llm_mentioned_surfaces.add(s.lower())
            for al in _normalize_aliases(item.get("aliases"), s):
                llm_mentioned_surfaces.add(al.lower())

    scanned_aliases_by_src: dict[str, list[str]] = {}
    confirmed_kb_srcs: set[str] = set()
    scanned_terms: list[dict[str, Any]] = []
    rare_terms: list[dict[str, Any]] = []
    if kb_data and full_text:
        scanned_terms = _scan_text_for_kb_terms(full_text, kb_data)
        rare_terms = _retrieve_rare_token_candidates(full_text, kb_data)
        for scanned in scanned_terms:
            src_low = scanned["src"].lower()
            als = list(scanned.get("aliases") or [])
            prev_als = scanned_aliases_by_src.setdefault(src_low, [])
            for al in als:
                al_str = str(al).strip()
                if not al_str:
                    continue
                if al_str.lower() not in {x.lower() for x in prev_als}:
                    prev_als.append(al_str)
                auth_entry = kb_entries.get(src_low)
                if auth_entry is not None:
                    auth_by_lower.setdefault(al_str.lower(), auth_entry)
                    _, al_st, _ = _tokenize_surface(al_str)
                    if al_st:
                        auth_by_stem.setdefault(tuple(al_st), auth_entry)

        for rc in rare_terms:
            sl = rc["src"].lower()
            m_spans = list(rc.get("matched_spans") or rc.get("aliases") or [])
            if sl in llm_mentioned_surfaces or any(
                str(sp).strip().lower() in llm_mentioned_surfaces for sp in m_spans
            ):
                confirmed_kb_srcs.add(sl)
                prev_als = scanned_aliases_by_src.setdefault(sl, [])
                auth_entry = kb_entries.get(sl)
                for sp in m_spans:
                    sp_str = str(sp).strip()
                    if not sp_str:
                        continue
                    if sp_str.lower() not in {x.lower() for x in prev_als}:
                        prev_als.append(sp_str)
                    if auth_entry is not None:
                        auth_by_lower.setdefault(sp_str.lower(), auth_entry)
                        _, sp_st, _ = _tokenize_surface(sp_str)
                        if sp_st:
                            auth_by_stem.setdefault(tuple(sp_st), auth_entry)

    # Pass 2: Ground unresolved Pass-1 LLM terms & un-echoed pre-filter spans via UnifiedNgramKBIndex + LLM disambiguation
    ngram_idx: UnifiedNgramKBIndex | None = (
        kb_data.get("ngram_index") if kb_data else None
    )
    if (
        run_llm_disambiguation
        and kb_data
        and ngram_idx is not None
        and len(kb_entries) >= 10
        and full_text
    ):
        text_lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

        def _find_context_sentence(span_str: str) -> str:
            sl = span_str.lower()
            for ln in text_lines:
                if sl in ln.lower():
                    return ln
            toks = [w for w in _tokenize_surface(span_str)[0] if len(w) >= 3]
            for ln in text_lines:
                ll = ln.lower()
                if toks and all(t in ll for t in toks):
                    return ln
            return ""

        unresolved_map: dict[str, dict[str, Any]] = {}
        auto_resolved: dict[str, dict[str, Any]] = {}

        for _, res in ordered_chunks:
            if not isinstance(res, dict):
                continue
            for item in res.get("terms") or []:
                if not isinstance(item, dict):
                    continue
                s = str(item.get("src") or "").strip()
                if not s:
                    continue
                _, st, _ = _tokenize_surface(s)
                st_tup = tuple(st) if st else ()
                if s.lower() in auth_by_lower or (st_tup and st_tup in auth_by_stem):
                    continue
                als = _normalize_aliases(item.get("aliases"), s)
                if any(a.lower() in auth_by_lower for a in als):
                    continue
                queries = [s, *als]
                cand_scores: dict[str, float] = {}
                cand_entries: dict[str, dict[str, Any]] = {}
                for q in queries:
                    for sc, ent in ngram_idx.query(q, top_k=3, min_score=0.43):
                        ek = ent["src"].lower()
                        if sc > cand_scores.get(ek, 0.0):
                            cand_scores[ek] = sc
                            cand_entries[ek] = ent
                if not cand_scores:
                    continue
                ranked = sorted(cand_scores.items(), key=lambda x: -x[1])[:3]
                best_ek, best_sc = ranked[0]
                second_sc = ranked[1][1] if len(ranked) > 1 else 0.0
                if best_sc >= 0.84 and (best_sc - second_sc) >= 0.12:
                    auto_resolved[s.lower()] = cand_entries[best_ek]
                    for a in als:
                        auto_resolved[a.lower()] = cand_entries[best_ek]
                else:
                    ctx = _find_context_sentence(
                        als[0] if als else s
                    ) or _find_context_sentence(s)
                    unresolved_map[s.lower()] = {
                        "span": s,
                        "aliases": als,
                        "sentence": ctx,
                        "candidates": [
                            {
                                "src": cand_entries[ek]["src"],
                                "tgt": cand_entries[ek]["tgt"],
                                "note": cand_entries[ek].get("note", ""),
                                "score": round(sc, 3),
                            }
                            for ek, sc in ranked
                        ],
                    }

        for rc in rare_terms:
            sl = rc["src"].lower()
            m_spans = list(rc.get("matched_spans") or rc.get("aliases") or [])
            if sl in llm_mentioned_surfaces or any(
                str(sp).strip().lower() in llm_mentioned_surfaces for sp in m_spans
            ):
                continue
            for sp in m_spans:
                sp_str = str(sp).strip()
                sp_low = sp_str.lower()
                if not sp_low or sp_low in auth_by_lower or sp_low in unresolved_map:
                    continue
                cands_list = [
                    {
                        "src": rc["src"],
                        "tgt": rc["tgt"],
                        "note": rc.get("note", ""),
                        "score": 0.75,
                    }
                ]
                for sc, ent in ngram_idx.query(sp_str, top_k=3, min_score=0.45):
                    if ent["src"].lower() != sl and len(cands_list) < 3:
                        cands_list.append(
                            {
                                "src": ent["src"],
                                "tgt": ent["tgt"],
                                "note": ent.get("note", ""),
                                "score": round(sc, 3),
                            }
                        )
                unresolved_map[sp_low] = {
                    "span": sp_str,
                    "aliases": [],
                    "sentence": _find_context_sentence(sp_str),
                    "candidates": cands_list,
                }

        for sp_low, ent in auto_resolved.items():
            ek = ent["src"].lower()
            confirmed_kb_srcs.add(ek)
            auth_by_lower.setdefault(sp_low, ent)
            prev_als = scanned_aliases_by_src.setdefault(ek, [])
            if sp_low != ek and sp_low not in {x.lower() for x in prev_als}:
                prev_als.append(sp_low)

        if unresolved_map:
            llm_decisions = _resolve_ambiguous_kb_candidates_with_llm(
                list(unresolved_map.values())
            )
            for sp_low, chosen_src in llm_decisions.items():
                ent = kb_entries.get(chosen_src.lower())
                if ent is None:
                    continue
                ek = ent["src"].lower()
                confirmed_kb_srcs.add(ek)
                auth_by_lower[sp_low] = ent
                prev_als = scanned_aliases_by_src.setdefault(ek, [])
                orig_span = unresolved_map.get(sp_low, {}).get("span", sp_low)
                if orig_span.lower() != ek and orig_span.lower() not in {
                    x.lower() for x in prev_als
                }:
                    prev_als.append(orig_span)
                for a in unresolved_map.get(sp_low, {}).get("aliases") or []:
                    al_low = a.lower()
                    auth_by_lower.setdefault(al_low, ent)
                    if al_low != ek and al_low not in {x.lower() for x in prev_als}:
                        prev_als.append(a)

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
        raw_aliases = list(aliases or [])
        if auth is None:
            for a in raw_aliases:
                auth = auth_by_lower.get(str(a).strip().lower())
                if auth is not None:
                    break
        extra_aliases: list[str] = []
        if auth is not None:
            canonical_low = auth["src"].lower()
            for a in raw_aliases:
                a_str = str(a).strip()
                if not a_str:
                    continue
                a_auth = auth_by_lower.get(a_str.lower())
                if a_auth is not None and a_auth["src"].lower() != canonical_low:
                    continue
                a_toks = _tokenize_surface(a_str)[0]
                if any(
                    len(tok) >= 5
                    and tok in auth_by_lower
                    and auth_by_lower[tok]["src"].lower() != canonical_low
                    for tok in a_toks
                ):
                    continue
                extra_aliases.append(a_str)
            if s.lower() != canonical_low:
                s_auth = auth_by_lower.get(s.lower())
                if s_auth is None or s_auth["src"].lower() == canonical_low:
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
        else:
            extra_aliases = raw_aliases

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

    for s_low in confirmed_kb_srcs:
        if s_low not in merged_by_src and s_low in kb_entries:
            ent = kb_entries[s_low]
            _upsert_term(
                ent["src"],
                ent["tgt"],
                str(ent.get("note") or ""),
                scanned_aliases_by_src.get(s_low, []),
                bool(ent.get("fix_source", False)),
                source="task",
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

    # Propagate 1-word OOV proper noun aliases to possessive multi-word entries in merged_by_src
    for k, entry in list(merged_by_src.items()):
        bw, _, _ = _tokenize_surface(entry["src"])
        if len(bw) == 1 and len(bw[0]) >= 5 and not _is_standard_english(bw[0]):
            w0 = bw[0]
            als = entry.get("aliases") or []
            if not als:
                continue
            for other_k, other_ent in merged_by_src.items():
                if other_k == k:
                    continue
                obw, _, _ = _tokenize_surface(other_ent["src"])
                if len(obw) >= 2 and obw[0] == w0:
                    rest = " ".join(obw[1:])
                    new_als = list(other_ent.get("aliases") or [])
                    for a in als:
                        cand_al = f"{a} {rest}"
                        if cand_al.lower() not in {x.lower() for x in new_als}:
                            new_als.append(cand_al)
                    other_ent["aliases"] = _normalize_aliases(new_als, other_ent["src"])

    _derive_in_video_shorthands(merged_by_src, full_text)

    return {"theme": combined_theme, "terms": list(merged_by_src.values())}


def _derive_in_video_shorthands(
    terms_map: dict[str, dict[str, Any]], full_text: str = ""
) -> set[str]:
    """Derive 1-word constituent shorthands for unique multi-word terms in the video and return all claimed lowercase surfaces."""
    tok_to_multi: dict[str, list[dict[str, Any]]] = {}
    protected_single: set[str] = set()
    for entry in terms_map.values():
        bw, _, _ = _tokenize_surface(entry["src"])
        if len(bw) == 1:
            if bw[0] not in _EXTRA_COMMON_WORDS:
                protected_single.add(bw[0])
        elif len(bw) >= 2:
            for w in set(bw):
                if len(w) >= 4 and w not in _KB_STOP_TOKENS:
                    tok_to_multi.setdefault(w, []).append(entry)

    if full_text:
        for w, ent_list in tok_to_multi.items():
            if len(ent_list) != 1 or w in protected_single:
                continue
            ent = ent_list[0]
            canon = ent["src"]
            cleaned = full_text
            for surf in (canon, *(ent.get("aliases") or [])):
                if surf:
                    cleaned = re.sub(re.escape(surf), " ", cleaned, flags=re.IGNORECASE)
            all_occ = re.findall(rf"\b{re.escape(w)}\b", cleaned, flags=re.IGNORECASE)
            if not all_occ:
                continue
            mid_cap = re.findall(
                rf"(?<![.!?,:;]\s)(?<![\n])\b{re.escape(w.capitalize())}\b", cleaned
            )
            cap_ratio = sum(1 for x in all_occ if x[0].isupper()) / len(all_occ)
            is_oov = not _is_standard_english(w)
            if is_oov or (mid_cap and cap_ratio >= 0.5):
                shorthand = mid_cap[0] if mid_cap else all_occ[0]
                cur_als = list(ent.get("aliases") or [])
                if shorthand.lower() not in {a.lower() for a in cur_als}:
                    cur_als.append(shorthand)
                    ent["aliases"] = _normalize_aliases(cur_als, canon)
                if w in terms_map and w in _EXTRA_COMMON_WORDS:
                    terms_map.pop(w, None)

    claimed: set[str] = set()
    for entry in terms_map.values():
        claimed.add(entry["src"].lower())
        for al in entry.get("aliases") or []:
            claimed.add(str(al).strip().lower())
    return claimed


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
            rare_cands = _retrieve_rare_token_candidates(chunk, kb_data)
            seen_cand_srcs = set(chunk_srcs)
            chunk_candidates = list(chunk_kb)
            for rc in rare_cands:
                rl = rc["src"].lower()
                if rl not in seen_cand_srcs:
                    seen_cand_srcs.add(rl)
                    chunk_candidates.append(rc)
                else:
                    for existing in chunk_candidates:
                        if existing["src"].lower() == rl:
                            existing["aliases"] = _normalize_aliases(
                                [
                                    *(existing.get("aliases") or []),
                                    *(rc.get("aliases") or []),
                                ],
                                existing["src"],
                            )
            context_kb = [
                t for t in video_kb_terms if t["src"].lower() not in chunk_srcs
            ]
            futures.append(
                executor.submit(
                    _extract_chunk_terms, idx, chunk, chunk_candidates, context_kb
                )
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
        chunk_results,
        custom_terms_list,
        kb_data=kb_data,
        full_text=full_text,
        run_llm_disambiguation=True,
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

        task_is_primary = bool(task_terms) and ws_mtime >= ext_mtime
        primary = task_terms if task_is_primary else custom_terms
        secondary = custom_terms if task_is_primary else task_terms

        merged_map: dict[str, dict[str, Any]] = {}
        for item in primary:
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

        claimed_by_task: set[str] = set()
        if task_is_primary:
            utts = ws.load_utterances() if hasattr(ws, "load_utterances") else []
            utt_text = "\n".join(
                str(u.get("src_punct") or u.get("src") or "").strip()
                for u in utts
                if str(u.get("src_punct") or u.get("src") or "").strip()
            )
            claimed_by_task = _derive_in_video_shorthands(merged_map, utt_text)

        for item in secondary:
            if not isinstance(item, dict):
                continue
            s = str(item.get("src") or "").strip()
            t = str(item.get("tgt") or "").strip()
            if not s or not t:
                continue
            k = s.lower()
            raw_als = _normalize_aliases(item.get("aliases"), s)
            als = (
                [a for a in raw_als if a.lower() not in claimed_by_task]
                if claimed_by_task
                else raw_als
            )
            if k not in merged_map:
                if k in claimed_by_task:
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

        norm_merged_map = _normalize_kb_entries(merged_map)
        all_entries = list(norm_merged_map.values())
        seen_all = set(norm_merged_map.keys()) | claimed_by_task
        for k, kb_entry in (kb_data.get("entries") or {}).items():
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

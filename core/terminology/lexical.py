from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from nltk.stem import SnowballStemmer
from rapidfuzz.distance import Levenshtein as _RFLevenshtein

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")
_UNICODE_SEP_RE = re.compile(r"[\u2010-\u2015\u2212\u2044/]+")
_SIGNIFICANT_SYMBOL_RE = re.compile(r"(^\.[A-Za-z0-9]+|[+#@$%&])")
_POSSESSIVE_SUFFIX_RE = re.compile(r"['’]s$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_INIT_VOWELS_RE = re.compile(r"^[aeiouy]+")
_SOFT_C_RE = re.compile(r"c(?=[eiy])")
_INNER_VOWELS_RE = re.compile(r"[aeiouyhw]")
_DEDUP_CHAR_RE = re.compile(r"(.)\1+")
_STEMMER = SnowballStemmer("english")

_CHAR_BIT: dict[str, int] = {
    ch: 1 << idx for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789")
}

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
    return int(_RFLevenshtein.distance(s1, s2))


@lru_cache(maxsize=65536)
def _phonetic_skeleton(word: str) -> str:
    s = _NON_ALNUM_RE.sub("", word.lower())
    if not s:
        return ""
    s = _INIT_VOWELS_RE.sub("a", s)
    s = _SOFT_C_RE.sub("s", s)
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
    rest = _INNER_VOWELS_RE.sub("", s[1:])
    raw = first + rest
    dedup: list[str] = []
    for ch in raw:
        if not dedup or dedup[-1] != ch:
            dedup.append(ch)
    return "".join(dedup)


@lru_cache(maxsize=65536)
def _phonetic_dt_collapsed(pa: str) -> str:
    if len(pa) < 3:
        return ""
    return pa[0] + _DEDUP_CHAR_RE.sub(r"\1", pa[1:].replace("d", "t"))


@lru_cache(maxsize=65536)
def _char_mask_and_dups(word: str) -> tuple[int, int]:
    mask = 0
    for ch in word:
        mask |= _CHAR_BIT.get(ch, 0)
    return mask, len(word) - mask.bit_count()


def _char_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 1.0 - _RFLevenshtein.distance(a, b) / max(len(a), len(b))


@lru_cache(maxsize=65536)
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
        da = _phonetic_dt_collapsed(pa)
        db = _phonetic_dt_collapsed(pb)
        p_sim = max(p_sim, _char_sim(da, db) * 0.96)
    p_weighted = (
        p_sim * 0.94
        if (len(pa) >= 2 and len(pb) >= 2 and pa[0] == pb[0])
        else p_sim * 0.80
    )
    return max(c_sim, p_weighted, 0.55 * c_sim + 0.45 * p_sim)


def _strip_word_punct(wl: str) -> tuple[str, str]:
    if "'" not in wl and "’" not in wl:
        return wl, wl
    bw = _POSSESSIVE_SUFFIX_RE.sub("", wl).replace("'", "").replace("’", "")
    rw = wl.replace("'", "").replace("’", "")
    return bw, rw


def _tokenize_surface(text: str) -> tuple[list[str], list[str], list[str]]:
    folded = _fold_ascii(text).lower()
    words = _WORD_RE.findall(folded)
    base_words: list[str] = []
    raw_words: list[str] = []
    for w in words:
        bw, rw = _strip_word_punct(w)
        if bw:
            base_words.append(bw)
            raw_words.append(rw)
    stems = [_stem_word(w) for w in base_words]
    return base_words, stems, raw_words


def _tokenize_and_normalize(text: str) -> tuple[list[str], list[str]]:
    base_words, stems, _ = _tokenize_surface(text)
    return base_words, stems


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

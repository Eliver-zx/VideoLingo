from __future__ import annotations

import re

_LEADING_PUNCT_RE = re.compile(
    r"^[\s，。、！!？?：:；;）)】\]》>…—\-·“”‘’\"'「」『』]+"
)
_QUOTE_BOOK_MARK_TABLE = str.maketrans({c: None for c in "《》〈〉“”‘’「」『』"})

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_DASH_COLON_SEMI_RE = re.compile(r"(?:——+|—+|--+|[：；;])+")
_NON_TIME_COLON_RE = re.compile(r"(?<!\d):(?!\d)")
_MULTI_COMMA_RE = re.compile(r"，+")
_COMMA_BEFORE_STOP_RE = re.compile(r"，([。？！、])")
_COMMA_AFTER_PUNCT_RE = re.compile(r"([，。？！、])，")

_CLAUSE_PUNCT = frozenset(",;:，、；：—")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,、；;：:。！？!?])|(?<=——)|(?<!—)(?<=—)(?!—)")
_BREAK_CONJUNCTIONS = frozenset(
    {
        "and",
        "but",
        "or",
        "so",
        "because",
        "since",
        "although",
        "though",
        "while",
        "when",
        "where",
        "which",
        "that",
        "who",
        "if",
        "unless",
        "until",
        "before",
        "after",
        "whether",
        "as",
        "yet",
        "nor",
        "to",
        "for",
        "with",
        "in",
        "on",
        "at",
        "from",
        "by",
        "about",
    }
)


def _normalize_book_marks(text: str) -> str:
    s = str(text).translate(_QUOTE_BOOK_MARK_TABLE)
    if s.isascii() or not _CJK_CHAR_RE.search(s):
        return s
    if '"' in s:
        s = s.replace('"', "")
    if "—" in s or "--" in s or "：" in s or "；" in s or ";" in s:
        s = _DASH_COLON_SEMI_RE.sub("，", s)
    if ":" in s:
        s = _NON_TIME_COLON_RE.sub("，", s)
    if "，" in s:
        if "，，" in s:
            s = _MULTI_COMMA_RE.sub("，", s)
        s = _COMMA_BEFORE_STOP_RE.sub(r"\1", s)
        s = _COMMA_AFTER_PUNCT_RE.sub(r"\1", s)
        s = s.strip("，")
    return s


def _char_weight(char: str) -> float:
    code = ord(char)
    if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
        return 1.75
    if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
        return 1.5
    if 0x0E00 <= code <= 0x0E7F:  # Thai
        return 1.0
    if 0xFF01 <= code <= 0xFF5E:  # full-width symbols
        return 1.75
    return 1.0


def calc_len(text: str) -> float:
    s = str(text)
    if s.isascii():
        return float(len(s))
    return sum(_char_weight(char) for char in s)


def needs_subtitle_split(
    src: str,
    tr: str,
    max_sub_length: float,
    target_sub_multiplier: float,
) -> tuple[bool, str]:
    src_s = str(src)
    tr_s = str(tr)
    tr_norm = _normalize_book_marks(tr_s)
    comfortable_w = max(38.0, float(max_sub_length) * 0.60)
    has_multi_clauses = (
        calc_len(tr_norm) * float(target_sub_multiplier) > comfortable_w
        and len([c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_norm) if c.strip()]) >= 2
    )
    should_split = (
        calc_len(src_s) > max_sub_length
        or calc_len(tr_s) * target_sub_multiplier > max_sub_length
        or has_multi_clauses
    )
    return should_split, tr_norm

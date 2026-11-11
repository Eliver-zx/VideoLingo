from __future__ import annotations

import re
from typing import Any

import pandas as pd

_QUOTE_BOOK_MARK_TABLE = str.maketrans({c: None for c in "《》〈〉“”‘’「」『』"})

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_DASH_TO_COMMA_RE = re.compile(r"\s*(?:——|—|--)\s*")
_COLON_SEMI_RE = re.compile(r"[：；]")
_CJK_DIGIT_SPACE_RE = re.compile(r"([\u4e00-\u9fff])\s+([\d\uff10-\uff19])")
_DIGIT_CJK_SPACE_RE = re.compile(r"([\d\uff10-\uff19])\s+([\u4e00-\u9fff])")
_CJK_LATIN_SPACE_RE = re.compile(r"([\u4e00-\u9fff])\s+([a-zA-Z])")
_LATIN_CJK_SPACE_RE = re.compile(r"([a-zA-Z])\s+([\u4e00-\u9fff])")
_CJK_CLAUSE_PUNCT_RE = re.compile(r"[，。、？！：；—]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NON_WORD_SPACE_RE = re.compile(r"[^\w\s]")


def _seconds_to_hmsm(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem_seconds = seconds % 60
    milliseconds = int(rem_seconds * 1000) % 1000
    return f"{hours:02d}:{minutes:02d}:{int(rem_seconds):02d},{milliseconds:03d}"


def convert_to_srt_format(start_time: float, end_time: float) -> str:
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    return f"{_seconds_to_hmsm(start_time)} --> {_seconds_to_hmsm(end_time)}"


def remove_punctuation(text: str) -> str:
    cleaned = _MULTI_SPACE_RE.sub(" ", text)
    cleaned = _NON_WORD_SPACE_RE.sub("", cleaned)
    return cleaned.strip()


def _strip_quotes_and_dashes(text: str) -> str:
    s = str(text).translate(_QUOTE_BOOK_MARK_TABLE)
    if not s.isascii() and _CJK_CHAR_RE.search(s) and '"' in s:
        s = s.replace('"', "")
    if "—" in s or "--" in s:
        s = _DASH_TO_COMMA_RE.sub("，", s)
    if "：" in s or "；" in s:
        s = _COLON_SEMI_RE.sub("，", s)
    return s


def format_subtitle_text(x: Any) -> str:
    """Format subtitle text: remove spaces between CJK and Latin/digits, strip quotes/dashes/colons, and replace CJK clause punctuation with spaces."""
    if pd.isna(x):
        return ""
    text = _strip_quotes_and_dashes(str(x))
    if not text.isascii():
        text = _CJK_DIGIT_SPACE_RE.sub(r"\1\2", text)
        text = _DIGIT_CJK_SPACE_RE.sub(r"\1\2", text)
        text = _CJK_LATIN_SPACE_RE.sub(r"\1\2", text)
        text = _LATIN_CJK_SPACE_RE.sub(r"\1\2", text)
        text = _CJK_CLAUSE_PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def clean_translation(x: Any) -> str:
    if pd.isna(x):
        return ""
    return _strip_quotes_and_dashes(str(x)).strip("。，、？！：；—").strip()


def generate_subtitle_string(df: pd.DataFrame, columns: list[str]) -> str:
    timestamps = df["timestamp"].tolist()
    col0 = [str(v).strip() for v in df[columns[0]].tolist()]
    if len(columns) > 1:
        col1 = [str(v).strip() for v in df[columns[1]].tolist()]
        return "".join(
            f"{i}\n{ts}\n{c0}\n{c1}\n\n"
            for i, (ts, c0, c1) in enumerate(zip(timestamps, col0, col1), start=1)
        ).strip()
    return "".join(
        f"{i}\n{ts}\n{c0}\n\n\n"
        for i, (ts, c0) in enumerate(zip(timestamps, col0), start=1)
    ).strip()

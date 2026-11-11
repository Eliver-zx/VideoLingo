import re
import unicodedata
from typing import Any

CHUNK_WORDS = 400
OVERLAP_WORDS = 60
SENTENCE_END_CHARS = ".?!。？！…।॥؟۔"
CLAUSE_PUNCT_CHARS = ",，;；:：、—–،؛"
PAUSE_MARKER = "‖"
LONG_PAUSE_MARKER = "‖‖"
ACOUSTIC_PAUSE_PROMPT_GAP = 0.40
ACOUSTIC_LONG_PAUSE_GAP = 1.00

_STRICT_TITLE_ABBREVS = set(
    "mr mrs ms dr prof sr jr st vs fig no vol rev est approx".split()
)
_ABBREVIATIONS = _STRICT_TITLE_ABBREVS | set("etc inc ltd corp co eg ie".split())

_STANDALONE_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_TRAILING_QUOTE_CHARS = "\"'”’）)]}"


def _is_word_char(ch: str) -> bool:
    if ord(ch) < 128:
        return ch.isalnum()
    if ch == PAUSE_MARKER:
        return False
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N") or cat in ("Mn", "Mc", "Me")


def _is_unspaced_script_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0x0E00 <= cp <= 0x0EFF
        or 0x1000 <= cp <= 0x109F
        or 0x1780 <= cp <= 0x17FF
    )


def _split_atomic_tokens(text: str) -> list[str]:
    """Split text into atomic tokens (each CJK/Thai char or spaced word + trailing punct)."""
    s = str(text)
    n = len(s)
    tokens: list[str] = []
    i = 0
    while i < n:
        if s[i].isspace() or s[i] == PAUSE_MARKER:
            i += 1
            continue
        start = i
        if _is_unspaced_script_char(s[i]):
            i += 1
            while i < n and unicodedata.category(s[i]) in ("Mn", "Mc", "Me"):
                i += 1
            while (
                i < n
                and not s[i].isspace()
                and s[i] != PAUSE_MARKER
                and not _is_word_char(s[i])
            ):
                i += 1
            tokens.append(s[start:i])
        elif _is_word_char(s[i]):
            while i < n and not s[i].isspace() and s[i] != PAUSE_MARKER:
                if _is_unspaced_script_char(s[i]):
                    break
                if _is_word_char(s[i]):
                    i += 1
                elif (
                    s[i] in ".':-"
                    and i + 1 < n
                    and _is_word_char(s[i + 1])
                    and not _is_unspaced_script_char(s[i + 1])
                ):
                    i += 1
                else:
                    break
            while (
                i < n
                and not s[i].isspace()
                and s[i] != PAUSE_MARKER
                and not _is_word_char(s[i])
            ):
                i += 1
            tokens.append(s[start:i])
        else:
            while (
                i < n
                and not s[i].isspace()
                and s[i] != PAUSE_MARKER
                and not _is_word_char(s[i])
            ):
                i += 1
            if tokens:
                tokens[-1] += s[start:i]
            else:
                tokens.append(s[start:i])
    return tokens


def _clean_word(text: str) -> str:
    s = str(text).lower()
    if s.isascii():
        if s.isalnum():
            return s
        return "".join(ch for ch in s if ch.isalnum())
    return "".join(ch for ch in s if _is_word_char(ch))


def _merge_standalone_punct_tokens(
    words: list[str],
    word_dicts: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]] | None]:
    """Attach standalone punctuation tokens onto the neighboring word so indices match words."""
    merged_words: list[str] = []
    merged_dicts: list[dict[str, Any]] | None = [] if word_dicts is not None else None
    has_dicts = (
        word_dicts is not None
        and merged_dicts is not None
        and len(word_dicts) == len(words)
    )
    pending_lead = ""
    pending_start: float | None = None

    for idx, raw in enumerate(words):
        w = str(raw).strip()
        wd = dict(word_dicts[idx]) if has_dicts and word_dicts is not None else None
        if not w:
            continue
        if _STANDALONE_PUNCT_RE.match(w) and not _clean_word(w):
            if merged_words:
                merged_words[-1] = merged_words[-1] + w
                if merged_dicts and wd is not None:
                    merged_dicts[-1]["text"] = merged_words[-1]
                    if "end" in wd:
                        merged_dicts[-1]["end"] = max(
                            float(merged_dicts[-1].get("end", 0.0)),
                            float(wd.get("end", 0.0)),
                        )
            else:
                pending_lead += w
                if (
                    pending_start is None
                    and wd is not None
                    and float(wd.get("start", 0.0)) > 0.0
                ):
                    pending_start = float(wd["start"])
        else:
            if pending_lead:
                w = pending_lead + w
                pending_lead = ""
            merged_words.append(w)
            if merged_dicts is not None and wd is not None:
                wd["text"] = w
                if pending_start is not None:
                    wd["start"] = pending_start
                    pending_start = None
                merged_dicts.append(wd)

    if pending_lead and merged_words:
        merged_words[-1] += pending_lead
        if merged_dicts:
            merged_dicts[-1]["text"] = merged_words[-1]
    return merged_words, merged_dicts


def _prepare_word_inputs(
    words: list[str] | list[dict[str, Any]],
    word_dicts: list[dict[str, Any]] | None = None,
) -> tuple[
    list[str],
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
    list[float] | None,
]:
    caller_dicts: list[dict[str, Any]] | None = word_dicts
    if isinstance(words[0], dict):
        dict_words: list[dict[str, Any]] = list(words)  # type: ignore[arg-type]
        caller_dicts = dict_words
        raw_word_strs = [str(w.get("text", w.get("word", ""))) for w in dict_words]
    else:
        raw_word_strs = [str(w) for w in words]

    cleaned_words, active_dicts = _merge_standalone_punct_tokens(
        raw_word_strs, caller_dicts
    )
    n_words = len(cleaned_words)
    word_gaps = (
        [
            round(
                max(
                    0.0,
                    float(active_dicts[i + 1].get("start", 0.0))
                    - float(active_dicts[i].get("end", 0.0)),
                ),
                3,
            )
            if i + 1 < n_words
            else 0.0
            for i in range(n_words)
        ]
        if active_dicts is not None and len(active_dicts) == n_words
        else None
    )
    return cleaned_words, active_dicts, caller_dicts, word_gaps


def _sync_word_dicts(
    active_dicts: list[dict[str, Any]] | None,
    caller_dicts: list[dict[str, Any]] | None,
    punct_by_word: dict[int, str],
    kept_spans: list[tuple[int, int]],
) -> None:
    if active_dicts is None:
        return
    for i, wd in enumerate(active_dicts):
        wd["idx"] = i
        wd["punct_after"] = punct_by_word.get(i, "")
    if active_dicts:
        active_dicts[0]["_sentence_spans"] = kept_spans
    if caller_dicts is None:
        return
    if len(caller_dicts) == len(active_dicts):
        for src_d, act_d in zip(caller_dicts, active_dicts):
            src_d["text"] = act_d.get("text", src_d.get("text", ""))
            src_d["end"] = act_d.get("end", src_d.get("end", 0.0))
            src_d["punct_after"] = act_d.get("punct_after", "")
        if caller_dicts:
            caller_dicts[0]["_sentence_spans"] = kept_spans
    else:
        caller_dicts[:] = active_dicts


def _is_true_sentence_end_on_word(
    word_str: str,
    punct_char: str,
    gap_after: float = 0.0,
    next_word: str | None = None,
) -> bool:
    """Return True if ``punct_char`` at the end of ``word_str`` is a real sentence boundary."""
    if not punct_char or punct_char not in SENTENCE_END_CHARS:
        return False
    if punct_char != ".":
        return True
    s = str(word_str).strip().rstrip(_TRAILING_QUOTE_CHARS)
    had_raw_dot = s.endswith(".")
    core = s[:-1] if had_raw_dot else s
    clean_core = _clean_word(core)
    if not clean_core:
        return False
    if clean_core in _STRICT_TITLE_ABBREVS:
        return False
    next_stripped = (
        str(next_word)
        .replace(PAUSE_MARKER, "")
        .lstrip(_TRAILING_QUOTE_CHARS + " \"'“‘（([{")
        if next_word
        else ""
    )
    next_is_upper = bool(next_stripped and next_stripped[0].isupper())
    if clean_core in _ABBREVIATIONS:
        return bool(gap_after >= 0.30 or next_is_upper)
    if "." in core:
        if not had_raw_dot:
            return True
        return bool(gap_after >= 0.30 or next_is_upper)
    if (
        len(clean_core) == 1
        and clean_core.isalpha()
        and core[-1:].isupper()
        and had_raw_dot
        and gap_after < 0.30
        and not next_is_upper
    ):
        return False
    return True


def _existing_sentence_ends(
    words: list[str],
    joiner: str = " ",
    word_gaps: list[float] | None = None,
) -> set[int]:
    """Fallback sentence-boundary detector when LLM punctuation fails."""
    ends: set[int] = set()
    for i, w in enumerate(words):
        s = str(w).strip().rstrip(_TRAILING_QUOTE_CHARS)
        if not s:
            continue
        gap_i = word_gaps[i] if (word_gaps is not None and i < len(word_gaps)) else 0.0
        next_w = words[i + 1] if i + 1 < len(words) else None
        if s[-1] in SENTENCE_END_CHARS and _is_true_sentence_end_on_word(
            s, s[-1], gap_after=gap_i, next_word=next_w
        ):
            if i + 1 < len(words):
                next_clean = _clean_word(words[i + 1])
                if next_clean and next_clean[0].isdigit() and s[-1] == ".":
                    continue
            ends.add(i)

    fallback_chunk = 35 if joiner == " " else 45
    last_end = -1
    for i, w in enumerate(words):
        gap = word_gaps[i] if (word_gaps is not None and i < len(word_gaps)) else 0.0
        s = str(w).strip().rstrip(_TRAILING_QUOTE_CHARS)
        has_clause_punct = bool(s and s[-1] in CLAUSE_PUNCT_CHARS)
        if i in ends:
            last_end = i
        elif (i - last_end >= 12 and gap >= 0.65) or (
            i - last_end >= 18 and has_clause_punct and gap >= 0.3
        ):
            ends.add(i)
            last_end = i
        elif i - last_end >= fallback_chunk:
            ends.add(i)
            last_end = i

    if words:
        ends.add(len(words) - 1)
    return ends


def build_overlapping_windows(
    total_words: int,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[tuple[int, int, int, int]]:
    """Build ``(w_start, w_end, own_start, own_end)`` windows with mid-overlap handoff."""
    if total_words <= 0:
        return []
    chunk_words = max(10, int(chunk_words))
    overlap_words = max(0, min(int(overlap_words), chunk_words // 2))
    step = max(1, chunk_words - overlap_words)

    raw_bounds: list[tuple[int, int]] = []
    start = 0
    while start < total_words:
        end = min(total_words, start + chunk_words)
        raw_bounds.append((start, end))
        if end >= total_words:
            break
        start += step

    n_win = len(raw_bounds)
    windows: list[tuple[int, int, int, int]] = []
    for idx, (w_start, w_end) in enumerate(raw_bounds):
        own_start = 0 if idx == 0 else (raw_bounds[idx - 1][1] + w_start) // 2
        own_end = (
            total_words if idx == n_win - 1 else (w_end + raw_bounds[idx + 1][0]) // 2
        )
        windows.append((w_start, w_end, own_start, own_end))
    return windows

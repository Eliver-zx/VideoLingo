from __future__ import annotations

import concurrent.futures
from difflib import SequenceMatcher
import os
from pathlib import Path
import re
import shutil
from typing import Any
import unicodedata

import pandas as pd

from core.prompts import get_punctuation_prompt
from core.task_runner import StopTask
from core.utils import (
    ask_gpt,
    check_cancel,
    get_joiner,
    get_source_language,
    join_words,
    load_key_or,
    rprint,
)
from core.utils.models import (
    _2_CLEANED_CHUNKS,
    _3_1_SPLIT_BY_NLP,
    _3_2_SPLIT_BY_MEANING,
    resolve_task_path,
)
from core.workspace import TaskWorkspace, get_active_workspace

CHUNK_WORDS = 400
OVERLAP_WORDS = 60
SENTENCE_END_CHARS = ".?!。？！…।॥؟۔"
CLAUSE_PUNCT_CHARS = ",，、;；:：—"
PAUSE_MARKER = "‖"
ACOUSTIC_PAUSE_PROMPT_GAP = 0.40
# Sentence completeness takes priority before translation; Step 5 (_5_split_sub.py)
# handles splitting translated sentences into short display cues.
# Only split unpunctuated runaway spans exceeding 120 words / 180 CJK chars and 25s.
LONG_SENTENCE_MAX_WORDS = 120
LONG_SENTENCE_MAX_CJK_CHARS = 180
LONG_SENTENCE_MAX_DURATION = 25.0
CLAUSE_SPLIT_MIN_GAP = 0.25

_SUBORDINATE_CLAUSE_STARTERS = {
    "because",
    "cuz",
    "cause",
    "since",
    "although",
    "though",
    "while",
    "whereas",
    "unless",
    "until",
    "if",
    "whether",
    "especially",
    "particularly",
    "specifically",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "whereby",
    "wherein",
    "meaning",
    "including",
    "excluding",
}
_DANGLING_TAIL_WORDS = {
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "into",
    "about",
    "like",
    "and",
    "or",
    "but",
    "nor",
    "because",
    "cuz",
    "if",
    "while",
    "when",
    "although",
    "though",
    "which",
    "who",
    "where",
    "than",
    "as",
}
_FILLER_TOKENS = {"uh", "um", "er", "ah"}

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "corp",
    "fig",
    "vol",
    "ch",
    "eg",
    "ie",
}
_DOMAIN_OR_EXT = {
    "com",
    "org",
    "net",
    "io",
    "ai",
    "dev",
    "gov",
    "edu",
    "js",
    "ts",
    "py",
    "rs",
    "go",
    "cpp",
    "json",
    "yaml",
    "yml",
    "md",
    "txt",
    "csv",
    "html",
    "css",
}


def _is_word_char(ch: str) -> bool:
    if ch == PAUSE_MARKER:
        return False
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N") or cat in ("Mn", "Mc", "Me")


def _clean_word(word: str) -> str:
    return "".join(ch for ch in str(word).lower() if _is_word_char(ch))


def _starts_with_upper(token: str | None) -> bool:
    if not token:
        return False
    stripped = str(token).replace(PAUSE_MARKER, "").lstrip("\"'“‘（([{ ")
    return bool(stripped and stripped[0].isupper())


def _is_sentence_end_punct(
    word: str,
    next_word: str | int | None = None,
    gap_after: float = 0.0,
) -> bool:
    """Return True if ``word`` (or ``word[next_word]`` when ``next_word`` is an int index)
    ends with a genuine sentence-ending punctuation mark.

    Excludes:
    - Decimal points inside numbers (``3.5``, ``v2.0``)
    - Dots inside acronyms or file extensions/domains (``U.S.A.``, ``node.js``)
    - Common honorifics/abbreviations (``Mr.``, ``Dr.``, ``vs.``, ``e.g.``)
    - Single-letter initials unless ``gap_after >= 0.30`` or ``next_word`` starts with an uppercase letter
    """
    if isinstance(next_word, int):
        text = str(word)
        idx = next_word
        if idx < 0 or idx >= len(text):
            return False
        ch = text[idx]
        if ch not in SENTENCE_END_CHARS:
            return False

        n = len(text)
        if idx + 1 < n and text[idx + 1] in SENTENCE_END_CHARS:
            return False

        if ch != ".":
            return True

        prev_ch = text[idx - 1] if idx > 0 else ""
        next_ch = text[idx + 1] if idx + 1 < n else ""

        if prev_ch.isdigit() and next_ch.isdigit():
            return False
        if next_ch and _is_word_char(next_ch):
            return False

        rest_after = text[idx + 1 :].replace(PAUSE_MARKER, " ").lstrip(" \"'”’）)]}")
        m_next = re.match(r"(\S+)", rest_after)
        following_word = m_next.group(1) if m_next else None

        left_slice = text[max(0, idx - 20) : idx]
        m_word = re.search(r"([A-Za-z0-9.]+)$", left_slice)
        if m_word:
            token = m_word.group(1)
            clean_tok = token.replace(".", "").lower()
            if clean_tok in _ABBREVIATIONS:
                return False
            if "." in token and len(clean_tok) >= 2:
                if gap_after >= 0.30 or _starts_with_upper(following_word):
                    return True
                return False
            if len(clean_tok) == 1 and clean_tok.isalpha():
                if gap_after >= 0.30 or _starts_with_upper(following_word):
                    return True
                return False

        return True

    # Word-token mode: _is_sentence_end_punct(word, next_word=None, gap_after=0.0)
    stripped = str(word).replace(PAUSE_MARKER, "").rstrip("\"'”’）)]}")
    if not stripped or stripped[-1] not in SENTENCE_END_CHARS:
        return False
    if stripped[-1] != ".":
        return True

    body = stripped[:-1]
    clean_tok = _clean_word(body)
    if not clean_tok or clean_tok in _ABBREVIATIONS:
        return False
    if "." in body and len(clean_tok) >= 2:
        if gap_after >= 0.30 or _starts_with_upper(next_word):
            return True
        return False
    if len(clean_tok) == 1 and clean_tok.isalpha():
        if gap_after >= 0.30 or _starts_with_upper(next_word):
            return True
        return False
    return True


def build_overlapping_windows(
    n_words: int,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[tuple[int, int, int, int]]:
    """Build overlapping word index windows ``(w_start, w_end, own_start, own_end)``."""
    if n_words <= 0:
        return []
    step = max(1, chunk_words - overlap_words)
    raw_windows: list[tuple[int, int]] = []
    start = 0
    while start < n_words:
        end = min(n_words, start + chunk_words)
        raw_windows.append((start, end))
        if end >= n_words:
            break
        start += step

    windows: list[tuple[int, int, int, int]] = []
    num_win = len(raw_windows)
    for idx, (w_start, w_end) in enumerate(raw_windows):
        own_start = 0 if idx == 0 else (raw_windows[idx - 1][1] + w_start) // 2
        own_end = (
            n_words if idx == num_win - 1 else (w_end + raw_windows[idx + 1][0]) // 2
        )
        windows.append((w_start, w_end, own_start, own_end))
    return windows


def _existing_sentence_ends(
    win_words: list[str],
    joiner: str = " ",
    word_gaps: list[float] | None = None,
) -> set[int]:
    """Fallback sentence-end word indices based on existing trailing punctuation."""
    ends: set[int] = set()
    n = len(win_words)
    for i, word in enumerate(win_words):
        next_w = win_words[i + 1] if i + 1 < n else None
        gap = word_gaps[i] if word_gaps and i < len(word_gaps) else 0.0
        if _is_sentence_end_punct(word, next_word=next_w, gap_after=gap):
            ends.add(i)
    return ends


def _verify_word_alignment(win_words: list[str], restored_text: str) -> float:
    """Verify character alignment ratio after stripping acoustic pause markers ``‖``."""
    cleaned_restored = str(restored_text).replace(PAUSE_MARKER, "")
    src_clean = "".join(_clean_word(w) for w in win_words)
    dst_clean = _clean_word(cleaned_restored)
    if not src_clean or not dst_clean:
        return 0.0
    return SequenceMatcher(None, src_clean, dst_clean, autojunk=False).ratio()


def _project_window_punctuation(
    win_words: list[str],
    restored_text: str,
    word_gaps: list[float] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Project both sentence-ending and clause punctuation from ``restored_text`` onto ``win_words``."""
    full_clean = ""
    char_to_word: list[int] = []
    for i, word in enumerate(win_words):
        cw = _clean_word(word)
        full_clean += cw
        char_to_word.extend([i] * len(cw))

    existing_ends = _existing_sentence_ends(win_words, word_gaps=word_gaps)
    existing_punct: dict[int, str] = {}
    for i, w in enumerate(win_words):
        stripped = str(w).replace(PAUSE_MARKER, "").rstrip("\"'”’）)]}")
        if stripped and (
            stripped[-1] in SENTENCE_END_CHARS or stripped[-1] in CLAUSE_PUNCT_CHARS
        ):
            if i in existing_ends or stripped[-1] in CLAUSE_PUNCT_CHARS:
                existing_punct[i] = stripped[-1]

    if not full_clean:
        return existing_ends, existing_punct

    text_str = str(restored_text).replace(PAUSE_MARKER, "")

    restored_clean = ""
    sent_end_at_rpos: dict[int, str] = {}
    clause_punct_at_rpos: dict[int, str] = {}
    for idx, ch in enumerate(text_str):
        if _is_word_char(ch):
            restored_clean += ch.lower()
        elif restored_clean:
            r_pos = len(restored_clean) - 1
            if _is_sentence_end_punct(text_str, idx):
                sent_end_at_rpos[r_pos] = ch
            elif ch in CLAUSE_PUNCT_CHARS and r_pos not in sent_end_at_rpos:
                clause_punct_at_rpos[r_pos] = ch

    if not restored_clean or (not sent_end_at_rpos and not clause_punct_at_rpos):
        return existing_ends, existing_punct

    matcher = SequenceMatcher(None, full_clean, restored_clean, autojunk=False)
    blocks = matcher.get_matching_blocks()

    r_to_f: dict[int, int] = {}
    for a, b, size in blocks:
        for offset in range(size):
            r_to_f[b + offset] = a + offset

    def _map_rpos_to_word(r_pos: int) -> int | None:
        f_pos = r_to_f.get(r_pos)
        if f_pos is None:
            for delta in range(1, 15):
                if r_pos - delta in r_to_f:
                    f_pos = r_to_f[r_pos - delta]
                    break
                if r_pos + delta in r_to_f:
                    f_pos = r_to_f[r_pos + delta]
                    break
        if f_pos is not None and 0 <= f_pos < len(char_to_word):
            return char_to_word[f_pos]
        return None

    end_word_indices: set[int] = set()
    # When the LLM returns restored punctuation, trust the LLM's punctuation placement
    # rather than carrying over raw ASR trailing periods that the LLM chose to remove.
    punct_by_word: dict[int, str] = {}

    for r_pos, ch in sorted(clause_punct_at_rpos.items()):
        w_idx = _map_rpos_to_word(r_pos)
        if w_idx is not None:
            punct_by_word[w_idx] = ch

    for r_pos, ch in sorted(sent_end_at_rpos.items()):
        w_idx = _map_rpos_to_word(r_pos)
        if w_idx is not None:
            # Verify single-letter initial with acoustic gap if applicable
            w_str = win_words[w_idx]
            clean_tok = _clean_word(w_str)
            if ch == "." and len(clean_tok) == 1 and clean_tok.isalpha():
                next_w = win_words[w_idx + 1] if w_idx + 1 < len(win_words) else None
                gap = word_gaps[w_idx] if word_gaps and w_idx < len(word_gaps) else 0.0
                if not _is_sentence_end_punct(
                    w_str + ".", next_word=next_w, gap_after=gap
                ):
                    continue
            end_word_indices.add(w_idx)
            punct_by_word[w_idx] = ch

    if not end_word_indices:
        return existing_ends, dict(existing_punct)
    return end_word_indices, punct_by_word


def project_sentence_ends(
    win_words: list[str],
    restored_text: str,
    word_gaps: list[float] | None = None,
) -> set[int]:
    """Project sentence-ending punctuation from ``restored_text`` onto ``win_words`` indices."""
    ends, _ = _project_window_punctuation(win_words, restored_text, word_gaps=word_gaps)
    return ends


def _merge_standalone_punct_tokens(
    words: list[str],
    word_dicts: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]] | None]:
    """Attach standalone punctuation tokens (e.g. ``['你好', '。']``) to the preceding word
    so word indices and char_to_word mappings never leave dangling leading punctuation."""
    merged: list[str] = []
    merged_dicts: list[dict[str, Any]] | None = [] if word_dicts is not None else None
    for idx, raw in enumerate(words):
        w = str(raw).strip('"').strip()
        wd = (
            word_dicts[idx]
            if (word_dicts is not None and idx < len(word_dicts))
            else None
        )
        if not w:
            continue
        if not _clean_word(w):
            if merged:
                merged[-1] = merged[-1] + w
                if merged_dicts and wd is not None:
                    merged_dicts[-1]["end"] = max(
                        float(merged_dicts[-1].get("end", 0.0)),
                        float(wd.get("end", 0.0)),
                    )
                    if w[-1] in SENTENCE_END_CHARS or w[-1] in CLAUSE_PUNCT_CHARS:
                        merged_dicts[-1]["punct_after"] = w[-1]
            continue
        merged.append(w)
        if merged_dicts is not None:
            merged_dicts.append(
                dict(wd) if wd is not None else {"text": w, "start": 0.0, "end": 0.0}
            )
    return merged, merged_dicts


def _normalize_slice_words(slice_words: list[str], default_period: str) -> list[str]:
    """Normalize trailing punctuation on words inside a single sentence slice:
    - Internal words have erroneous sentence-ending periods/question/exclamation marks stripped
      (unless they are abbreviations like ``Mr.``, dotted initialisms like ``U.S.``, or decimals like ``3.5``).
    - The final word has trailing commas/colons replaced with a proper sentence-ending mark.
    """
    if not slice_words:
        return []
    out = list(slice_words)
    for i in range(len(out) - 1):
        w = out[i]
        if w and w[-1] in SENTENCE_END_CHARS:
            clean_tok = _clean_word(w)
            if w[-1] == "." and (
                clean_tok in _ABBREVIATIONS
                or "." in w[:-1]
                or (len(clean_tok) == 1 and clean_tok.isalpha())
            ):
                continue
            out[i] = w.rstrip(SENTENCE_END_CHARS)

    last = out[-1].rstrip(CLAUSE_PUNCT_CHARS)
    if not last:
        last = out[-1]
    if (
        last
        and last[-1] not in SENTENCE_END_CHARS
        and not last.endswith(('"', "'", "”", "’", ")", "）", "]"))
    ):
        last = last + default_period
    out[-1] = last
    return out


def _strip_prompt_punct(word: str) -> str:
    """Strip ASR trailing sentence/clause punctuation before sending to the LLM so the model
    makes syntactic boundary decisions cleanly while keeping abbreviations and decimals intact."""
    s = str(word).strip()
    if not s:
        return s
    if s[-1] in CLAUSE_PUNCT_CHARS:
        return s[:-1] or s
    if s[-1] in SENTENCE_END_CHARS:
        clean_tok = _clean_word(s[:-1])
        if s[-1] == "." and (
            clean_tok in _ABBREVIATIONS
            or "." in s[:-1]
            or (len(clean_tok) == 1 and clean_tok.isalpha())
        ):
            return s
        return s.rstrip(SENTENCE_END_CHARS) or s
    return s


def _build_window_segment_with_pauses(
    win_words: list[str],
    joiner: str,
    word_gaps: list[float] | None = None,
) -> str:
    prompt_words = [_strip_prompt_punct(w) for w in win_words]
    if not word_gaps or not prompt_words:
        return join_words(prompt_words, joiner)
    pieces: list[str] = []
    for i, w in enumerate(prompt_words):
        pieces.append(w)
        if (
            i + 1 < len(prompt_words)
            and i < len(word_gaps)
            and word_gaps[i] >= ACOUSTIC_PAUSE_PROMPT_GAP
        ):
            pieces.append(PAUSE_MARKER)
    return join_words(pieces, joiner)


def _restore_window_punctuation(
    win_idx: int,
    win_words: list[str],
    joiner: str,
    language: str,
    word_gaps: list[float] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Call LLM to restore punctuation for one window and return local sentence-end indices and punct map."""
    check_cancel()
    raw_segment = _build_window_segment_with_pauses(
        win_words, joiner, word_gaps=word_gaps
    )
    prompt = get_punctuation_prompt(raw_segment, language=language)

    def valid_punct(resp):
        if not isinstance(resp, dict) or not isinstance(resp.get("restored"), str):
            return {
                "status": "error",
                "message": "Missing 'restored' string in response",
            }
        restored_val = resp["restored"].replace(PAUSE_MARKER, "").strip()
        if not restored_val:
            return {"status": "error", "message": "Empty 'restored' string"}
        if win_words:
            ratio = _verify_word_alignment(win_words, restored_val)
            if ratio < 0.80:
                return {
                    "status": "error",
                    "message": f"Restored text alignment ratio too low ({ratio:.2f} < 0.80)",
                }
        return {"status": "success", "message": "ok"}

    try:
        resp = ask_gpt(
            prompt,
            resp_type="json",
            valid_def=valid_punct,
            log_title=f"punct_restore_{win_idx}",
        )
        restored_text = resp["restored"].replace(PAUSE_MARKER, "")
        return _project_window_punctuation(
            win_words, restored_text, word_gaps=word_gaps
        )
    except StopTask:
        raise
    except Exception as exc:
        rprint(
            f"[yellow]⚠️ Punctuation restoration fallback on chunk {win_idx} ({exc}); "
            "using existing punctuation.[/yellow]"
        )
        ends = _existing_sentence_ends(win_words, joiner=joiner, word_gaps=word_gaps)
        fallback_punct: dict[int, str] = {}
        for i, w in enumerate(win_words):
            s = str(w).rstrip("\"'”’）)]}")
            if s and (s[-1] in SENTENCE_END_CHARS or s[-1] in CLAUSE_PUNCT_CHARS):
                fallback_punct[i] = s[-1]
        return ends, fallback_punct


def _split_long_span_at_clauses(
    start_idx: int,
    end_idx: int,
    cleaned_words: list[str],
    punct_by_word: dict[int, str],
    word_dicts: list[dict[str, Any]] | None,
    word_gaps: list[float] | None,
) -> list[tuple[int, int]]:
    """Split overly long sentences (>45 words or >60 CJK chars and >8.0s) at internal clause punctuation."""
    n_span_words = end_idx - start_idx + 1
    cjk_chars = sum(
        len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", cleaned_words[k]))
        for k in range(start_idx, end_idx + 1)
    )
    duration = (
        float(word_dicts[end_idx].get("end", 0.0))
        - float(word_dicts[start_idx].get("start", 0.0))
        if word_dicts and end_idx < len(word_dicts)
        else 0.0
    )
    is_overlong = (
        n_span_words > LONG_SENTENCE_MAX_WORDS
        or cjk_chars > LONG_SENTENCE_MAX_CJK_CHARS
    ) and (duration > LONG_SENTENCE_MAX_DURATION if word_dicts else True)
    if not is_overlong or n_span_words < 6:
        return [(start_idx, end_idx)]

    clause_candidates: list[int] = []
    for k in range(start_idx + 2, end_idx - 1):
        p = punct_by_word.get(k, "")
        w_tail = cleaned_words[k][-1] if cleaned_words[k] else ""
        if (p and p in CLAUSE_PUNCT_CHARS) or (w_tail and w_tail in CLAUSE_PUNCT_CHARS):
            next_tok = _clean_word(cleaned_words[k + 1]) if k + 1 <= end_idx else ""
            if next_tok in _SUBORDINATE_CLAUSE_STARTERS:
                continue
            clause_candidates.append(k)

    if not clause_candidates:
        return [(start_idx, end_idx)]

    mid = (start_idx + end_idx) / 2.0
    pause_candidates = [
        k
        for k in clause_candidates
        if word_gaps is not None
        and k < len(word_gaps)
        and word_gaps[k] >= CLAUSE_SPLIT_MIN_GAP
    ]
    pool = pause_candidates if pause_candidates else clause_candidates
    best_k = min(pool, key=lambda k: abs(k - mid))

    left_spans = _split_long_span_at_clauses(
        start_idx, best_k, cleaned_words, punct_by_word, word_dicts, word_gaps
    )
    right_spans = _split_long_span_at_clauses(
        best_k + 1, end_idx, cleaned_words, punct_by_word, word_dicts, word_gaps
    )
    return left_spans + right_spans


def _stitch_incomplete_clause_spans(
    spans: list[tuple[int, int]],
    cleaned_words: list[str],
    punct_by_word: dict[int, str],
    word_gaps: list[float] | None,
) -> list[tuple[int, int]]:
    """Merge false sentence breaks where the preceding span ends with a dangling token or
    is a short fronted adverbial clause, or the following span starts with a subordinate clause
    without an acoustic pause."""
    if len(spans) <= 1:
        return spans

    stitched: list[tuple[int, int]] = [spans[0]]
    for cur_s, cur_e in spans[1:]:
        prev_s, prev_e = stitched[-1]
        gap = word_gaps[prev_e] if (word_gaps and prev_e < len(word_gaps)) else 0.0
        prev_last_tok = _clean_word(cleaned_words[prev_e])

        first_tok = _clean_word(cleaned_words[cur_s])
        non_filler_idx = cur_s
        while (
            non_filler_idx < cur_e
            and _clean_word(cleaned_words[non_filler_idx]) in _FILLER_TOKENS
        ):
            non_filler_idx += 1
        effective_first_tok = _clean_word(cleaned_words[non_filler_idx])

        prev_head_toks = [
            _clean_word(cleaned_words[k])
            for k in range(prev_s, min(prev_e + 1, prev_s + 4))
        ]
        prev_has_fronted_sub = any(
            t in {"while", "although", "though", "when", "whenever", "if", "unless"}
            for t in prev_head_toks
        )
        prev_has_internal_comma = any(
            bool(punct_by_word.get(k))
            and (punct_by_word.get(k) or "") in CLAUSE_PUNCT_CHARS
            for k in range(prev_s + 2, prev_e)
        )
        cur_has_internal_comma = any(
            bool(punct_by_word.get(k))
            and (punct_by_word.get(k) or "") in CLAUSE_PUNCT_CHARS
            for k in range(cur_s + 2, cur_e)
        )

        combined_words = cur_e - prev_s + 1
        should_stitch = False
        if (
            prev_last_tok in _DANGLING_TAIL_WORDS
            and combined_words <= LONG_SENTENCE_MAX_WORDS
        ):
            should_stitch = True
        elif (
            first_tok in _FILLER_TOKENS
            and effective_first_tok
            in {"you", "a", "the", "like", "for", "with", "on", "in", "to", "because"}
            and gap < 0.45
            and combined_words <= 60
        ):
            should_stitch = True
        elif (
            effective_first_tok in _SUBORDINATE_CLAUSE_STARTERS
            and gap < 0.45
            and combined_words <= 60
            and (
                not cur_has_internal_comma
                or effective_first_tok
                in {
                    "because",
                    "cuz",
                    "whereas",
                    "which",
                    "especially",
                    "including",
                    "meaning",
                }
            )
        ):
            should_stitch = True
        elif (
            prev_has_fronted_sub
            and not prev_has_internal_comma
            and (prev_e - prev_s + 1) <= 25
            and combined_words <= 60
            and gap < 0.45
        ):
            should_stitch = True

        if should_stitch:
            punct_by_word[prev_e] = ","
            stitched[-1] = (prev_s, cur_e)
        else:
            stitched.append((cur_s, cur_e))

    return stitched


def restore_and_split_words(
    words: list[str] | list[dict[str, Any]],
    language: str | None = None,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    word_dicts: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Restore punctuation across ``words`` in overlapping batches and split by sentence ends."""
    if not words:
        return []

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
    if not cleaned_words:
        return []

    lang = language or get_source_language()
    joiner = get_joiner(lang)
    n_words = len(cleaned_words)

    word_gaps: list[float] | None = None
    if active_dicts is not None and len(active_dicts) == n_words:
        word_gaps = []
        for i in range(n_words):
            if i + 1 < n_words:
                gap = max(
                    0.0,
                    float(active_dicts[i + 1].get("start", 0.0))
                    - float(active_dicts[i].get("end", 0.0)),
                )
            else:
                gap = 0.0
            word_gaps.append(round(gap, 3))

    windows = build_overlapping_windows(
        n_words, chunk_words=chunk_words, overlap_words=overlap_words
    )
    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))

    global_end_indices: set[int] = set()
    global_punct_by_word: dict[int, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_win = {
            executor.submit(
                _restore_window_punctuation,
                idx,
                cleaned_words[w_start:w_end],
                joiner,
                lang,
                word_gaps[w_start:w_end] if word_gaps is not None else None,
            ): (w_start, w_end, own_start, own_end)
            for idx, (w_start, w_end, own_start, own_end) in enumerate(windows)
        }
        try:
            for future in concurrent.futures.as_completed(future_to_win):
                check_cancel()
                w_start, _w_end, own_start, own_end = future_to_win[future]
                res = future.result()
                if isinstance(res, tuple):
                    local_ends, local_punct = res
                else:
                    local_ends, local_punct = res, {}
                for local_idx, p_ch in local_punct.items():
                    global_idx = w_start + local_idx
                    if own_start <= global_idx < own_end:
                        global_punct_by_word[global_idx] = p_ch
                for local_idx in local_ends:
                    global_idx = w_start + local_idx
                    if own_start <= global_idx < own_end:
                        global_end_indices.add(global_idx)
        except BaseException:
            for f in future_to_win:
                f.cancel()
            raise

    global_end_indices.add(n_words - 1)
    sorted_ends = sorted(idx for idx in global_end_indices if 0 <= idx < n_words)

    default_period = "。" if joiner == "" else "."
    raw_spans: list[tuple[int, int]] = []
    start_idx = 0
    for end_idx in sorted_ends:
        if end_idx >= start_idx:
            raw_spans.append((start_idx, end_idx))
            start_idx = end_idx + 1

    raw_spans = _stitch_incomplete_clause_spans(
        raw_spans, cleaned_words, global_punct_by_word, word_gaps
    )

    final_spans: list[tuple[int, int]] = []
    for s_idx, e_idx in raw_spans:
        final_spans.extend(
            _split_long_span_at_clauses(
                s_idx,
                e_idx,
                cleaned_words,
                global_punct_by_word,
                active_dicts,
                word_gaps,
            )
        )

    for _, e_idx in final_spans:
        if (
            e_idx not in global_punct_by_word
            or global_punct_by_word[e_idx] in CLAUSE_PUNCT_CHARS
        ):
            w_end_str = cleaned_words[e_idx].rstrip("\"'”’）)]}")
            if w_end_str and w_end_str[-1] in SENTENCE_END_CHARS:
                global_punct_by_word[e_idx] = w_end_str[-1]
            else:
                global_punct_by_word[e_idx] = default_period

    if active_dicts is not None:
        for i, wd in enumerate(active_dicts):
            wd["idx"] = i
            wd["punct_after"] = global_punct_by_word.get(i, "")
        if caller_dicts is not None:
            if len(caller_dicts) == len(active_dicts):
                for src_d, act_d in zip(caller_dicts, active_dicts):
                    src_d["text"] = act_d.get("text", src_d.get("text", ""))
                    src_d["end"] = act_d.get("end", src_d.get("end", 0.0))
                    src_d["punct_after"] = act_d.get("punct_after", "")
            else:
                caller_dicts[:] = active_dicts

    sentences: list[str] = []
    for s_idx, e_idx in final_spans:
        slice_words = list(cleaned_words[s_idx : e_idx + 1])
        if not slice_words:
            continue
        # Attach restored internal clause commas onto words (replacing any raw ASR period that the LLM downgraded)
        for rel_i in range(len(slice_words) - 1):
            abs_i = s_idx + rel_i
            p_ch = global_punct_by_word.get(abs_i, "")
            if p_ch and p_ch in CLAUSE_PUNCT_CHARS:
                w_cur = slice_words[rel_i]
                if w_cur and w_cur[-1] in SENTENCE_END_CHARS:
                    clean_tok = _clean_word(w_cur)
                    if not (
                        w_cur[-1] == "."
                        and (
                            clean_tok in _ABBREVIATIONS
                            or "." in w_cur[:-1]
                            or (len(clean_tok) == 1 and clean_tok.isalpha())
                        )
                    ):
                        w_cur = w_cur.rstrip(SENTENCE_END_CHARS)
                if (
                    w_cur
                    and w_cur[-1] not in CLAUSE_PUNCT_CHARS
                    and w_cur[-1] not in SENTENCE_END_CHARS
                ):
                    slice_words[rel_i] = w_cur + p_ch
        end_p = global_punct_by_word.get(e_idx, "")
        sent_end_punct = (
            end_p if (end_p and end_p in SENTENCE_END_CHARS) else default_period
        )
        norm_words = _normalize_slice_words(slice_words, sent_end_punct)
        sentence = join_words(norm_words, joiner).strip()
        sentence = sentence.lstrip("，,。.:;；：?!？！… ")
        if sentence:
            sentences.append(sentence)

    return sentences


def _build_utterances_from_sentences(
    sentences: list[str],
    word_dicts: list[dict[str, Any]],
    joiner: str,
) -> list[dict[str, Any]]:
    """Map ``sentences`` onto ``word_dicts`` to build ``Utterance`` records with exact ``word_span``."""
    if not sentences or not word_dicts:
        return []

    full_clean = ""
    char_to_word: list[int] = []
    for w_idx, wd in enumerate(word_dicts):
        cw = _clean_word(wd.get("text", wd.get("word", "")))
        full_clean += cw
        char_to_word.extend([w_idx] * len(cw))

    utterances: list[dict[str, Any]] = []
    cursor = 0
    n_words = len(word_dicts)

    for u_idx, sent in enumerate(sentences):
        sc = _clean_word(sent)
        if sc and full_clean:
            pos = full_clean.find(sc, cursor)
            if pos != -1:
                w_start = char_to_word[pos]
                w_end = char_to_word[pos + len(sc) - 1] + 1
                cursor = pos + len(sc)
            else:
                w_start = char_to_word[min(cursor, len(char_to_word) - 1)]
                end_char = min(len(char_to_word) - 1, cursor + max(1, len(sc)) - 1)
                w_end = max(w_start + 1, char_to_word[end_char] + 1)
                cursor = min(len(char_to_word), end_char + 1)
        else:
            w_start = min(u_idx, n_words - 1)
            w_end = min(n_words, w_start + 1)

        start_sec = round(float(word_dicts[w_start].get("start", 0.0)), 3)
        end_sec = round(float(word_dicts[w_end - 1].get("end", start_sec)), 3)
        if w_end < n_words:
            gap_after = round(
                max(0.0, float(word_dicts[w_end].get("start", end_sec)) - end_sec),
                3,
            )
        else:
            gap_after = 0.0
        src_raw = join_words(
            [
                str(w.get("text", w.get("word", ""))).strip()
                for w in word_dicts[w_start:w_end]
            ],
            joiner,
        ).strip()
        utterances.append(
            {
                "id": u_idx,
                "word_span": [w_start, w_end],
                "start": start_sec,
                "end": end_sec,
                "gap_after": gap_after,
                "src_raw": src_raw,
                "src": sent,
            }
        )
    return utterances


def split_by_spacy():
    nlp_path = resolve_task_path(_3_1_SPLIT_BY_NLP)
    meaning_path = resolve_task_path(_3_2_SPLIT_BY_MEANING)
    cleaned_path = resolve_task_path(_2_CLEANED_CHUNKS)

    if os.path.exists(nlp_path) and os.path.exists(meaning_path):
        rprint(
            f"[yellow]⚠️ Files <{nlp_path}> and <{meaning_path}> already exist, "
            "skip <split_by_spacy> step.[/yellow]"
        )
        return
    if os.path.exists(nlp_path) and not os.path.exists(meaning_path):
        shutil.copyfile(nlp_path, meaning_path)
        return

    language = get_source_language()
    joiner = get_joiner(language)
    rprint(
        f"[blue]🔍 Splitting sentences by LLM-restored punctuation (language: {language})...[/blue]"
    )

    ws = (
        get_active_workspace()
        if str(_2_CLEANED_CHUNKS).startswith("output/")
        else TaskWorkspace(
            Path(cleaned_path).parent.parent
            if Path(cleaned_path).parent.name == "log"
            else Path(cleaned_path).parent
        )
    )

    word_dicts: list[dict[str, Any]] = []
    if ws.words_path.is_file() and (
        not os.path.exists(cleaned_path)
        or os.path.getmtime(ws.words_path) >= os.path.getmtime(cleaned_path)
    ):
        word_dicts = ws.load_words()

    if not word_dicts:
        chunks = pd.read_excel(cleaned_path)
        for _idx, row in chunks.iterrows():
            raw_txt = str(row.get("text", "")).strip('"').strip()
            if not raw_txt:
                continue
            start_raw = row.get("start")
            end_raw = row.get("end")
            spk_raw = row.get("speaker_id")
            word_dicts.append(
                {
                    "idx": len(word_dicts),
                    "text": raw_txt,
                    "punct_after": "",
                    "start": 0.0
                    if start_raw is None or bool(pd.isna(start_raw))
                    else float(start_raw),
                    "end": 0.0
                    if end_raw is None or bool(pd.isna(end_raw))
                    else float(end_raw),
                    "speaker": None
                    if spk_raw is None or bool(pd.isna(spk_raw))
                    else spk_raw,
                }
            )

    words = [str(w.get("text", "")).strip('"').strip() for w in word_dicts]
    sentences = restore_and_split_words(words, language=language, word_dicts=word_dicts)

    ws.save_words(word_dicts)
    utterances = _build_utterances_from_sentences(sentences, word_dicts, joiner)
    ws.save_utterances(utterances)

    os.makedirs(os.path.dirname(nlp_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(meaning_path) or ".", exist_ok=True)
    output_text = "\n".join(sentences) + ("\n" if sentences else "")
    with open(nlp_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    with open(meaning_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    rprint(
        f"[green]💾 {len(sentences)} sentences saved to → `{nlp_path}` "
        f"and `{meaning_path}`[/green]"
    )


if __name__ == "__main__":
    split_by_spacy()

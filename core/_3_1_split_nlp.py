import concurrent.futures
import difflib
import re
import unicodedata
from typing import Any

from core.concurrency import ContextThreadPoolExecutor
from core.prompts import get_punctuation_prompt
from core.utils import (
    ask_gpt,
    check_cancel,
    get_joiner,
    get_source_language,
    join_words,
    load_key_or,
    log_event,
    report_progress,
)
from core.workspace import get_active_workspace

CHUNK_WORDS = 400
OVERLAP_WORDS = 60
SENTENCE_END_CHARS = ".?!。？！…।॥؟۔"
CLAUSE_PUNCT_CHARS = ",，;；:：、—–،؛"
PAUSE_MARKER = "‖"
LONG_PAUSE_MARKER = "‖‖"
ACOUSTIC_PAUSE_PROMPT_GAP = 0.40
ACOUSTIC_LONG_PAUSE_GAP = 1.00
CLAUSE_STITCH_MAX_GAP = 0.35
CLAUSE_SPLIT_MIN_GAP = 0.25
LONG_SENTENCE_MAX_WORDS = 120
LONG_SENTENCE_MAX_CJK_CHARS = 180
LONG_SENTENCE_MAX_DURATION = 25.0
STITCH_MAX_COMBINED_WORDS = 60

_STRICT_TITLE_ABBREVS = set(
    "mr mrs ms dr prof sr jr st vs fig no vol rev est approx".split()
)
_ABBREVIATIONS = _STRICT_TITLE_ABBREVS | set("etc inc ltd corp co eg ie".split())
_FRONTED_CLAUSE_STARTERS = set(
    "while although though even because cuz cause since unless until whereas "
    "whenever wherever whether if when as after before once".split()
)
_CLAUSE_CONJUNCTIONS = _FRONTED_CLAUSE_STARTERS | set(
    "which who whom whose where especially particularly specifically including "
    "excluding meaning".split()
)
_DISCOURSE_LEADINS = set("so and but now well ok okay hey yeah yes no".split())
_DANGLING_TAILS = _CLAUSE_CONJUNCTIONS | set(
    "that and but or nor so yet for to of in on at by with from into onto upon about above "
    "below under over between among through during without within against toward towards "
    "a an the my your his her its our their this these those is are was were be "
    "been being have has had do does did can could will would shall should may might must".split()
)
_FILLER_TOKENS = set("uh um er ah hmm uhh umm".split())

_STANDALONE_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_TRAILING_QUOTE_CHARS = "\"'”’）)]}"


def _is_word_char(ch: str) -> bool:
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
    return "".join(ch for ch in str(text).lower() if _is_word_char(ch))


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


def _extract_trailing_punct(token: str) -> str:
    """Extract the strongest trailing punctuation mark from ``token``."""
    stripped = str(token).strip().rstrip(_TRAILING_QUOTE_CHARS)
    if not stripped:
        return ""
    tail_punct = ""
    i = len(stripped) - 1
    while i >= 0 and (
        stripped[i] in SENTENCE_END_CHARS or stripped[i] in CLAUSE_PUNCT_CHARS
    ):
        ch = stripped[i]
        if ch in SENTENCE_END_CHARS:
            tail_punct = ch
            break
        if not tail_punct:
            tail_punct = ch
        i -= 1
    return tail_punct


def _tokenize_words_for_alignment(
    words: list[str],
) -> tuple[list[str], list[int], list[bool]]:
    """Expand ``words`` into atomic clean tokens with ``(clean_tok, word_idx, is_word_tail)``."""
    clean_toks: list[str] = []
    word_indices: list[int] = []
    is_tails: list[bool] = []
    for w_idx, w in enumerate(words):
        sub_matches = _split_atomic_tokens(str(w))
        valid_subs = [(_clean_word(m), m) for m in sub_matches if _clean_word(m)]
        if not valid_subs:
            cw = _clean_word(w)
            if cw:
                valid_subs = [(cw, str(w))]
        for sub_i, (cw, _) in enumerate(valid_subs):
            clean_toks.append(cw)
            word_indices.append(w_idx)
            is_tails.append(sub_i == len(valid_subs) - 1)
    return clean_toks, word_indices, is_tails


def _tokenize_restored_text(
    restored_text: str,
) -> tuple[list[str], list[str], list[str]]:
    """Tokenize ``restored_text`` into ``(clean_tokens, trailing_puncts, raw_tokens)``."""
    clean_toks: list[str] = []
    puncts: list[str] = []
    raw_toks: list[str] = []
    for raw_tok in _split_atomic_tokens(str(restored_text)):
        p_ch = _extract_trailing_punct(raw_tok)
        cw = _clean_word(raw_tok)
        if cw:
            clean_toks.append(cw)
            puncts.append(p_ch)
            raw_toks.append(raw_tok)
        elif p_ch and puncts:
            if p_ch in SENTENCE_END_CHARS or not puncts[-1]:
                puncts[-1] = p_ch
            raw_toks[-1] += raw_tok
    return clean_toks, puncts, raw_toks


def _verify_word_alignment(win_words: list[str], restored_text: str) -> float:
    """Return token-level similarity ratio in [0.0, 1.0] between ``win_words`` and ``restored_text``."""
    src_toks, _, _ = _tokenize_words_for_alignment(win_words)
    res_toks, _, _ = _tokenize_restored_text(restored_text)
    if not src_toks:
        return 1.0
    if not res_toks:
        return 0.0
    matcher = difflib.SequenceMatcher(None, src_toks, res_toks, autojunk=False)
    matched = sum(size for _, _, size in matcher.get_matching_blocks())
    return matched / len(src_toks)


def _project_window_punctuation(
    win_words: list[str],
    restored_text: str,
    word_gaps: list[float] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Project punctuation from ``restored_text`` onto ``win_words`` using token-level alignment."""
    if not win_words or not restored_text:
        return set(), {}

    src_toks, tok_to_word, _ = _tokenize_words_for_alignment(win_words)
    res_toks, res_puncts, res_raws = _tokenize_restored_text(restored_text)
    if not src_toks or not res_toks:
        return set(), {}

    matcher = difflib.SequenceMatcher(None, src_toks, res_toks, autojunk=False)
    res_to_src_tok: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                res_to_src_tok[j1 + offset] = i1 + offset
        elif tag == "replace" and (i2 > i1) and (j2 > j1):
            src_span = i2 - i1
            res_span = j2 - j1
            if src_span == res_span:
                for offset in range(res_span):
                    res_to_src_tok[j1 + offset] = i1 + offset
            else:
                for r_off in range(res_span - 1):
                    mapped = i1 + min(
                        src_span - 1, int((r_off + 1) * src_span / res_span) - 1
                    )
                    res_to_src_tok[j1 + r_off] = max(i1, mapped)
                # Multi-word alias/contraction replacement always anchors its final token to i2 - 1
                res_to_src_tok[j2 - 1] = i2 - 1

    end_indices: set[int] = set()
    punct_by_word: dict[int, str] = {}
    last_w_idx = len(win_words) - 1

    for r_idx, p_ch in enumerate(res_puncts):
        if not p_ch or r_idx not in res_to_src_tok:
            continue
        s_tok_idx = res_to_src_tok[r_idx]
        w_idx = tok_to_word[s_tok_idx]
        # Avoid splitting inside a multi-character word when a replace opcode hits an internal subtoken
        if s_tok_idx + 1 < len(tok_to_word) and tok_to_word[s_tok_idx + 1] == w_idx:
            continue

        prev_p = punct_by_word.get(w_idx, "")
        if p_ch in SENTENCE_END_CHARS or not prev_p:
            punct_by_word[w_idx] = p_ch

        gap_w = (
            word_gaps[w_idx]
            if (word_gaps is not None and w_idx < len(word_gaps))
            else 0.0
        )
        next_res_raw = (
            res_raws[r_idx + 1]
            if r_idx + 1 < len(res_raws)
            else (win_words[w_idx + 1] if w_idx + 1 < len(win_words) else None)
        )
        if p_ch in SENTENCE_END_CHARS and _is_true_sentence_end_on_word(
            win_words[w_idx],
            p_ch,
            gap_after=gap_w,
            next_word=next_res_raw,
        ):
            if w_idx == last_w_idx:
                raw_last = str(win_words[w_idx]).strip().rstrip(_TRAILING_QUOTE_CHARS)
                had_orig_end = bool(
                    raw_last
                    and raw_last[-1] in SENTENCE_END_CHARS
                    and _is_true_sentence_end_on_word(
                        win_words[w_idx], raw_last[-1], gap_after=gap_w
                    )
                )
                if not had_orig_end and gap_w < ACOUSTIC_PAUSE_PROMPT_GAP:
                    continue
            end_indices.add(w_idx)

    return end_indices, punct_by_word


def _strip_prompt_punct(word_str: str) -> str:
    """Strip unreliable raw ASR punctuation while keeping decimal and abbreviation dots."""
    s = str(word_str).strip()
    if not s:
        return s
    while s and (s[-1] in SENTENCE_END_CHARS or s[-1] in CLAUSE_PUNCT_CHARS):
        if s[-1] == "." and not _is_true_sentence_end_on_word(s, "."):
            break
        s = s[:-1]
    return s or str(word_str).strip()


def _build_window_prompt_text(
    win_words: list[str],
    joiner: str,
    word_gaps: list[float] | None = None,
) -> str:
    """Join ``win_words`` with tiered acoustic pause markers (`‖` >= 0.4s, `‖‖` >= 1.0s)."""
    pieces: list[str] = []
    n = len(win_words)
    for i, w in enumerate(win_words):
        pieces.append(_strip_prompt_punct(w))
        if word_gaps is not None and i < n - 1 and i < len(word_gaps):
            gap = word_gaps[i]
            if gap >= ACOUSTIC_LONG_PAUSE_GAP:
                pieces.append(LONG_PAUSE_MARKER)
            elif gap >= ACOUSTIC_PAUSE_PROMPT_GAP:
                pieces.append(PAUSE_MARKER)
    return join_words(pieces, joiner)


def _restore_window_punctuation(
    win_idx: int,
    win_words: list[str],
    joiner: str,
    language: str,
    word_gaps: list[float] | None = None,
) -> tuple[set[int], dict[int, str]]:
    """Call LLM to restore punctuation for one window and return ``(end_indices, punct_by_word)``."""
    raw_text = _build_window_prompt_text(win_words, joiner, word_gaps=word_gaps)
    prompt = get_punctuation_prompt(raw_text, language=language)

    def valid_punct(resp):
        if not isinstance(resp, dict) or "restored" not in resp:
            return {"status": "error", "message": "Missing `restored` key in JSON"}
        restored_val = str(resp.get("restored", "")).strip()
        if not restored_val:
            return {"status": "error", "message": "`restored` text is empty"}
        if len(win_words) >= 5:
            cleaned_restored = restored_val.replace(LONG_PAUSE_MARKER, "").replace(
                PAUSE_MARKER, ""
            )
            ratio = _verify_word_alignment(win_words, cleaned_restored)
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
        restored_text = (
            str(resp["restored"])
            .replace(LONG_PAUSE_MARKER, "")
            .replace(PAUSE_MARKER, "")
        )
        return _project_window_punctuation(
            win_words, restored_text, word_gaps=word_gaps
        )
    except Exception as exc:
        if type(exc).__name__ == "StopTask":
            raise
        log_event(
            f"标点恢复窗口 #{win_idx + 1} 降级为原标点 ({exc})",
            tag="warn",
            label="NLP",
            level="warn",
        )
        ends = _existing_sentence_ends(win_words, joiner=joiner, word_gaps=word_gaps)
        fallback_punct: dict[int, str] = {}
        for i, w in enumerate(win_words):
            s = str(w).rstrip(_TRAILING_QUOTE_CHARS)
            if s and (s[-1] in SENTENCE_END_CHARS or s[-1] in CLAUSE_PUNCT_CHARS):
                fallback_punct[i] = s[-1]
        return ends, fallback_punct


def _load_known_multiword_aliases(
    extra_terms: list[dict[str, Any]] | None = None,
) -> list[list[str]]:
    """Collect multi-word token sequences from workspace and custom/domain glossaries."""
    term_entries: list[dict[str, Any]] = []
    if extra_terms:
        term_entries.extend(extra_terms)
    try:
        ws = get_active_workspace()
        if ws.has_terms():
            term_entries.extend(ws.load_terms().get("terms") or [])
    except Exception:
        pass
    try:
        from core._4_1_summarize import load_custom_terms

        term_entries.extend(load_custom_terms())
    except Exception:
        pass

    seen: set[tuple[str, ...]] = set()
    patterns: list[list[str]] = []
    for entry in term_entries:
        if not isinstance(entry, dict):
            continue
        candidates = [entry.get("src"), *(entry.get("aliases") or [])]
        for cand in candidates:
            if not cand:
                continue
            toks, _, _ = _tokenize_words_for_alignment([str(cand)])
            if len(toks) >= 2:
                key = tuple(toks)
                if key not in seen:
                    seen.add(key)
                    patterns.append(toks)
    return patterns


def _protect_multiword_alias_spans(
    cleaned_words: list[str],
    end_indices: set[int],
    punct_by_word: dict[int, str],
    extra_terms: list[dict[str, Any]] | None = None,
) -> None:
    """Prevent sentence breaks inside matched multi-word glossary terms or aliases."""
    patterns = _load_known_multiword_aliases(extra_terms)
    if not patterns or len(cleaned_words) < 2:
        return
    word_clean = [_clean_word(w) for w in cleaned_words]
    n = len(word_clean)
    for pat in patterns:
        m = len(pat)
        if m > n:
            continue
        for start in range(n - m + 1):
            if word_clean[start : start + m] == pat:
                for internal_idx in range(start, start + m - 1):
                    end_indices.discard(internal_idx)
                    p_internal = punct_by_word.get(internal_idx, "")
                    if p_internal and p_internal in SENTENCE_END_CHARS:
                        punct_by_word.pop(internal_idx, None)


def _stitch_incomplete_clause_spans(
    spans: list[tuple[int, int]],
    cleaned_words: list[str],
    punct_by_word: dict[int, str],
    word_gaps: list[float] | None,
) -> list[tuple[int, int]]:
    """Merge adjacent spans when separated by a short acoustic pause and subordinate/dangling syntax."""
    if len(spans) <= 1:
        return spans

    comma_char = (
        "，"
        if any(
            re.search(r"[\u4e00-\u9fff]", cleaned_words[i])
            for i in range(min(20, len(cleaned_words)))
        )
        else ","
    )

    def _first_meaningful_word(s_idx: int, e_idx: int) -> str:
        for k in range(s_idx, min(e_idx + 1, s_idx + 4)):
            w = _clean_word(cleaned_words[k])
            if w and w not in _FILLER_TOKENS:
                return w
        return _clean_word(cleaned_words[s_idx])

    def _has_fronted_starter(s_idx: int, e_idx: int) -> bool:
        for k in range(s_idx, min(e_idx + 1, s_idx + 4)):
            w = _clean_word(cleaned_words[k])
            if not w or w in _FILLER_TOKENS or w in _DISCOURSE_LEADINS:
                continue
            return w in _FRONTED_CLAUSE_STARTERS
        return False

    stitched: list[tuple[int, int]] = [spans[0]]
    for nxt_s, nxt_e in spans[1:]:
        cur_s, cur_e = stitched[-1]
        cur_len = cur_e - cur_s + 1
        nxt_len = nxt_e - nxt_s + 1
        combined_len = nxt_e - cur_s + 1
        gap = (
            word_gaps[cur_e]
            if (word_gaps is not None and cur_e < len(word_gaps))
            else 0.0
        )

        cur_last = _clean_word(cleaned_words[cur_e])
        nxt_first = _first_meaningful_word(nxt_s, nxt_e)
        end_p = punct_by_word.get(cur_e, "")

        has_internal_comma = any(
            (pk := punct_by_word.get(k, "")) and pk in CLAUSE_PUNCT_CHARS
            for k in range(cur_s, cur_e)
        )
        nxt_has_internal_comma = any(
            (pk := punct_by_word.get(k, "")) and pk in CLAUSE_PUNCT_CHARS
            for k in range(nxt_s, nxt_e)
        )
        should_merge = False
        if combined_len <= STITCH_MAX_COMBINED_WORDS and end_p not in (
            "?",
            "!",
            "？",
            "！",
        ):
            if cur_last in _DANGLING_TAILS or cur_last in _FILLER_TOKENS:
                if gap <= 0.65:
                    should_merge = True
            elif (
                _has_fronted_starter(cur_s, cur_e)
                and cur_len <= 18
                and not has_internal_comma
                and gap <= CLAUSE_STITCH_MAX_GAP
            ):
                should_merge = True
            elif (
                nxt_first in _CLAUSE_CONJUNCTIONS
                and nxt_len <= 20
                and gap <= CLAUSE_STITCH_MAX_GAP
                and not (
                    nxt_first in _FRONTED_CLAUSE_STARTERS and nxt_has_internal_comma
                )
            ):
                should_merge = True

        if should_merge:
            punct_by_word[cur_e] = comma_char
            stitched[-1] = (cur_s, nxt_e)
        else:
            stitched.append((nxt_s, nxt_e))

    return stitched


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
            if next_tok in _CLAUSE_CONJUNCTIONS:
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

    return _split_long_span_at_clauses(
        start_idx, best_k, cleaned_words, punct_by_word, word_dicts, word_gaps
    ) + _split_long_span_at_clauses(
        best_k + 1, end_idx, cleaned_words, punct_by_word, word_dicts, word_gaps
    )


def _normalize_slice_words(slice_words: list[str], end_punct: str) -> list[str]:
    """Strip spurious mid-sentence periods while preserving abbreviations and decimals."""
    out = list(slice_words)
    n = len(out)
    for i, w in enumerate(out):
        s = str(w).strip()
        if not s:
            continue
        is_last = i == n - 1
        if not is_last:
            stripped = s.rstrip(_TRAILING_QUOTE_CHARS)
            if stripped and stripped[-1] in SENTENCE_END_CHARS:
                if _is_true_sentence_end_on_word(s, stripped[-1]):
                    quote_tail = s[len(stripped) :]
                    s = stripped[:-1].rstrip(".?!。？！") + quote_tail
        else:
            stripped = s.rstrip(_TRAILING_QUOTE_CHARS)
            quote_tail = s[len(stripped) :]
            if stripped and stripped[-1] in CLAUSE_PUNCT_CHARS:
                stripped = stripped[:-1]
            if stripped and stripped[-1] not in SENTENCE_END_CHARS:
                s = stripped + end_punct + quote_tail
            elif not stripped:
                s = end_punct
        out[i] = s
    return out


def restore_and_split_words(
    words: list[str] | list[dict[str, Any]],
    language: str | None = None,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    word_dicts: list[dict[str, Any]] | None = None,
    terms: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Restore punctuation across ``words`` in overlapping windows and split into sentences."""
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
        word_gaps = [
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

    windows = build_overlapping_windows(
        n_words, chunk_words=chunk_words, overlap_words=overlap_words
    )
    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))

    global_end_indices: set[int] = set()
    global_punct_by_word: dict[int, str] = {}
    total_win = max(1, len(windows))
    report_progress(
        0.05,
        f"标点恢复 0/{total_win}",
        f"正在使用 LLM 恢复标点与句子边界（共 {total_win} 个滑动窗口）",
        tag="nlp",
        label="NLP",
    )

    with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_win = {
            executor.submit(
                _restore_window_punctuation,
                idx,
                cleaned_words[w_start:w_end],
                joiner,
                lang,
                word_gaps[w_start:w_end] if word_gaps is not None else None,
            ): (idx, w_start, w_end, own_start, own_end)
            for idx, (w_start, w_end, own_start, own_end) in enumerate(windows)
        }
        try:
            done_win = 0
            for future in concurrent.futures.as_completed(future_to_win):
                check_cancel()
                win_idx, w_start, _w_end, own_start, own_end = future_to_win[future]
                res = future.result()
                done_win += 1
                report_progress(
                    0.05 + 0.85 * (done_win / total_win),
                    f"标点恢复 {done_win}/{total_win}",
                    f"完成标点恢复窗口 #{win_idx + 1:02d} ({done_win}/{total_win})",
                    tag="nlp",
                    label="NLP",
                )
                local_ends, local_punct = res if isinstance(res, tuple) else (res, {})
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

    _protect_multiword_alias_spans(
        cleaned_words, global_end_indices, global_punct_by_word, extra_terms=terms
    )
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
            w_end_str = cleaned_words[e_idx].rstrip(_TRAILING_QUOTE_CHARS)
            if w_end_str and w_end_str[-1] in SENTENCE_END_CHARS:
                global_punct_by_word[e_idx] = w_end_str[-1]
            else:
                global_punct_by_word[e_idx] = default_period

    sentences: list[str] = []
    kept_spans: list[tuple[int, int]] = []
    for s_idx, e_idx in final_spans:
        slice_words = list(cleaned_words[s_idx : e_idx + 1])
        if not slice_words:
            continue
        for rel_i in range(len(slice_words) - 1):
            abs_i = s_idx + rel_i
            p_ch = global_punct_by_word.get(abs_i, "")
            if p_ch and p_ch in CLAUSE_PUNCT_CHARS:
                w_cur = slice_words[rel_i]
                if w_cur and w_cur[-1] in SENTENCE_END_CHARS:
                    if _is_true_sentence_end_on_word(w_cur, w_cur[-1]):
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
        sentence = join_words(norm_words, joiner).strip().lstrip("，,。.:;；：?!？！… ")
        if sentence:
            sentences.append(sentence)
            kept_spans.append((s_idx, e_idx + 1))

    if active_dicts is not None:
        for i, wd in enumerate(active_dicts):
            wd["idx"] = i
            wd["punct_after"] = global_punct_by_word.get(i, "")
        if active_dicts:
            active_dicts[0]["_sentence_spans"] = kept_spans
        if caller_dicts is not None:
            if len(caller_dicts) == len(active_dicts):
                for src_d, act_d in zip(caller_dicts, active_dicts):
                    src_d["text"] = act_d.get("text", src_d.get("text", ""))
                    src_d["end"] = act_d.get("end", src_d.get("end", 0.0))
                    src_d["punct_after"] = act_d.get("punct_after", "")
                if caller_dicts:
                    caller_dicts[0]["_sentence_spans"] = kept_spans
            else:
                caller_dicts[:] = active_dicts

    return sentences


def _build_utterances_from_sentences(
    sentences: list[str],
    word_dicts: list[dict[str, Any]],
    joiner: str,
    spans: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Build ``Utterance`` records directly from precomputed ``spans`` or token alignment."""
    if not sentences or not word_dicts:
        return []

    n_words = len(word_dicts)
    resolved_spans = spans
    if resolved_spans is None and isinstance(
        word_dicts[0].get("_sentence_spans"), list
    ):
        cand_spans = word_dicts[0].pop("_sentence_spans", None)
        if isinstance(cand_spans, list) and len(cand_spans) == len(sentences):
            resolved_spans = cand_spans

    if resolved_spans is None or len(resolved_spans) != len(sentences):
        full_clean = ""
        char_to_word: list[int] = []
        for w_idx, wd in enumerate(word_dicts):
            cw = _clean_word(wd.get("text", wd.get("word", "")))
            full_clean += cw
            char_to_word.extend([w_idx] * len(cw))
        resolved_spans = []
        cursor = 0
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
            resolved_spans.append((w_start, w_end))

    utterances: list[dict[str, Any]] = []
    for u_idx, (sent, (w_start, w_end)) in enumerate(zip(sentences, resolved_spans)):
        w_start = max(0, min(int(w_start), n_words - 1))
        w_end = max(w_start + 1, min(int(w_end), n_words))
        start_sec = round(float(word_dicts[w_start].get("start", 0.0)), 3)
        end_sec = round(float(word_dicts[w_end - 1].get("end", start_sec)), 3)
        gap_after = (
            round(
                max(0.0, float(word_dicts[w_end].get("start", end_sec)) - end_sec),
                3,
            )
            if w_end < n_words
            else 0.0
        )
        src_raw = join_words(
            [
                str(w.get("text", w.get("word", "")))
                .strip()
                .strip(',.;:?!，。；：？！"“”')
                for w in word_dicts[w_start:w_end]
                if str(w.get("text", w.get("word", "")))
                .strip()
                .strip(',.;:?!，。；：？！"“”')
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
                "src_punct": sent,
                "src": sent,
            }
        )
    return utterances


def split_by_spacy():
    ws = get_active_workspace()

    if ws.has_utterances():
        report_progress(
            1.0,
            "复用已断句结果",
            "工作区已存在断句记录，跳过标点与断句阶段",
            tag="nlp",
            label="NLP",
        )
        return

    language = get_source_language()
    joiner = get_joiner(language)
    report_progress(
        0.02,
        f"准备标点恢复 ({language})…",
        f"开始执行 LLM 标点恢复与语义断句 (语言: {language})",
        tag="nlp",
        label="NLP",
    )

    word_dicts: list[dict[str, Any]] = ws.load_words()
    words = [str(w.get("text", "")).strip('"').strip() for w in word_dicts]
    sentences = restore_and_split_words(words, language=language, word_dicts=word_dicts)

    spans = (
        word_dicts[0].pop("_sentence_spans", None)
        if word_dicts and isinstance(word_dicts[0].get("_sentence_spans"), list)
        else None
    )
    ws.save_words(word_dicts)
    utterances = _build_utterances_from_sentences(
        sentences, word_dicts, joiner, spans=spans
    )
    ws.save_utterances(utterances)

    report_progress(
        1.0,
        f"共切分 {len(sentences)} 句",
        f"标点恢复与断句完成，共生成 {len(sentences)} 条语句",
        tag="nlp",
        label="NLP",
    )


if __name__ == "__main__":
    split_by_spacy()

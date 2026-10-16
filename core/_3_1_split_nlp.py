from __future__ import annotations

import concurrent.futures
import os
import re
import shutil
from difflib import SequenceMatcher

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
)

CHUNK_WORDS = 400
OVERLAP_WORDS = 30
SENTENCE_END_CHARS = ".?!。？！…"

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
    "no",
    "vol",
    "ch",
    "sec",
    "eg",
    "ie",
    "us",
    "uk",
    "eu",
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


def _clean_word(word: str) -> str:
    return re.sub(r"[^\w]", "", str(word).lower())


def _is_sentence_end_punct(text: str, idx: int) -> bool:
    """Return True if ``text[idx]`` is a genuine sentence-ending punctuation mark.

    Excludes:
    - Decimal points inside numbers (``3.5``, ``v2.0``)
    - Dots inside acronyms or file extensions/domains (``U.S.A.``, ``node.js``)
    - Common honorifics/abbreviations (``Mr.``, ``Dr.``, ``vs.``, ``e.g.``)
    - Contiguous ellipsis or repeated marks before the final mark (``...`` only fires on the last char)
    """
    ch = text[idx]
    if ch not in SENTENCE_END_CHARS:
        return False

    n = len(text)
    # In a run of consecutive sentence-end marks (e.g. "..." or "?!"), only fire on the last one
    if idx + 1 < n and text[idx + 1] in SENTENCE_END_CHARS:
        return False

    if ch in "?!。？！…":
        return True

    # ch == '.'
    prev_ch = text[idx - 1] if idx > 0 else ""
    next_ch = text[idx + 1] if idx + 1 < n else ""

    # Decimal point or middle of token without space (e.g., "3.5", "node.js", "a.b")
    if prev_ch.isdigit() and next_ch.isdigit():
        return False
    if next_ch and re.match(r"[\w]", next_ch):
        return False

    # Extract preceding word/token (including dots for U.S. / e.g.)
    left_slice = text[max(0, idx - 20) : idx]
    m_word = re.search(r"([A-Za-z0-9.]+)$", left_slice)
    if m_word:
        token = m_word.group(1)
        clean_tok = token.replace(".", "").lower()
        if clean_tok in _ABBREVIATIONS:
            return False
        # Single-letter initial like "J. K. Rowling" or "U.S." followed by space + capital letter
        if len(clean_tok) == 1 and clean_tok.isalpha():
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


def _existing_sentence_ends(win_words: list[str], joiner: str = " ") -> set[int]:
    """Fallback sentence-end word indices based on existing trailing punctuation."""
    ends: set[int] = set()
    for i, word in enumerate(win_words):
        stripped = str(word).rstrip("\"'”’）)]}")
        if not stripped or stripped[-1] not in SENTENCE_END_CHARS:
            continue
        if stripped[-1] == ".":
            clean_tok = _clean_word(stripped)
            if clean_tok in _ABBREVIATIONS or (
                len(clean_tok) == 1 and clean_tok.isalpha()
            ):
                continue
        ends.add(i)
    return ends


def project_sentence_ends(win_words: list[str], restored_text: str) -> set[int]:
    """Project sentence-ending punctuation from ``restored_text`` onto ``win_words`` indices."""
    full_clean = ""
    char_to_word: list[int] = []
    for i, word in enumerate(win_words):
        cw = _clean_word(word)
        full_clean += cw
        char_to_word.extend([i] * len(cw))

    if not full_clean:
        return _existing_sentence_ends(win_words)

    text_str = str(restored_text)
    restored_clean = ""
    punct_after_clean_pos: set[int] = set()
    for idx, ch in enumerate(text_str):
        if re.match(r"[\w]", ch):
            restored_clean += ch.lower()
        elif restored_clean and _is_sentence_end_punct(text_str, idx):
            punct_after_clean_pos.add(len(restored_clean) - 1)

    if not restored_clean or not punct_after_clean_pos:
        return _existing_sentence_ends(win_words)

    matcher = SequenceMatcher(None, full_clean, restored_clean, autojunk=False)
    blocks = matcher.get_matching_blocks()

    r_to_f: dict[int, int] = {}
    for a, b, size in blocks:
        for offset in range(size):
            r_to_f[b + offset] = a + offset

    end_word_indices: set[int] = set()
    for r_pos in sorted(punct_after_clean_pos):
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
            end_word_indices.add(char_to_word[f_pos])

    return end_word_indices or _existing_sentence_ends(win_words)


def _merge_standalone_punct_tokens(words: list[str]) -> list[str]:
    """Attach standalone punctuation tokens (e.g. ``['你好', '。']``) to the preceding word
    so word indices and char_to_word mappings never leave dangling leading punctuation."""
    merged: list[str] = []
    for raw in words:
        w = str(raw).strip('"').strip()
        if not w:
            continue
        if not _clean_word(w):
            if merged:
                merged[-1] = merged[-1] + w
            continue
        merged.append(w)
    return merged


def _normalize_slice_words(slice_words: list[str], default_period: str) -> list[str]:
    """Normalize trailing punctuation on words inside a single sentence slice:
    - Internal words have erroneous sentence-ending periods/question/exclamation marks stripped
      (unless they are abbreviations like ``Mr.`` or decimals like ``3.5``).
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
                or (len(clean_tok) == 1 and clean_tok.isalpha())
            ):
                continue
            out[i] = w.rstrip(SENTENCE_END_CHARS)

    last = out[-1].rstrip("，,、；;：:")
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


def _restore_window_punctuation(
    win_idx: int,
    win_words: list[str],
    joiner: str,
    language: str,
) -> set[int]:
    """Call LLM to restore punctuation for one window and return local sentence-end indices."""
    check_cancel()
    raw_segment = join_words(win_words, joiner)
    prompt = get_punctuation_prompt(raw_segment, language=language)

    def valid_punct(resp):
        if not isinstance(resp, dict) or not isinstance(resp.get("restored"), str):
            return {
                "status": "error",
                "message": "Missing 'restored' string in response",
            }
        if not resp["restored"].strip():
            return {"status": "error", "message": "Empty 'restored' string"}
        return {"status": "success", "message": "ok"}

    try:
        resp = ask_gpt(
            prompt,
            resp_type="json",
            valid_def=valid_punct,
            log_title=f"punct_restore_{win_idx}",
        )
        restored_text = resp["restored"]
        return project_sentence_ends(win_words, restored_text)
    except StopTask:
        raise
    except Exception as exc:
        rprint(
            f"[yellow]⚠️ Punctuation restoration fallback on chunk {win_idx} ({exc}); "
            "using existing punctuation.[/yellow]"
        )
        return _existing_sentence_ends(win_words, joiner=joiner)


def restore_and_split_words(
    words: list[str],
    language: str | None = None,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    """Restore punctuation across ``words`` in overlapping batches and split by sentence ends."""
    cleaned_words = _merge_standalone_punct_tokens(words)
    if not cleaned_words:
        return []

    lang = language or get_source_language()
    joiner = get_joiner(lang)
    n_words = len(cleaned_words)
    windows = build_overlapping_windows(
        n_words, chunk_words=chunk_words, overlap_words=overlap_words
    )
    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))

    global_end_indices: set[int] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_win = {
            executor.submit(
                _restore_window_punctuation,
                idx,
                cleaned_words[w_start:w_end],
                joiner,
                lang,
            ): (w_start, w_end, own_start, own_end)
            for idx, (w_start, w_end, own_start, own_end) in enumerate(windows)
        }
        try:
            for future in concurrent.futures.as_completed(future_to_win):
                check_cancel()
                w_start, _w_end, own_start, own_end = future_to_win[future]
                local_ends = future.result()
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

    sentences: list[str] = []
    start_idx = 0
    default_period = "。" if joiner == "" else "."
    for end_idx in sorted_ends:
        slice_words = cleaned_words[start_idx : end_idx + 1]
        start_idx = end_idx + 1
        if not slice_words:
            continue
        norm_words = _normalize_slice_words(slice_words, default_period)
        sentence = join_words(norm_words, joiner).strip()
        sentence = sentence.lstrip("，,。.:;；：?!？！… ")
        if sentence:
            sentences.append(sentence)

    return sentences


def split_by_spacy():
    if os.path.exists(_3_1_SPLIT_BY_NLP) and os.path.exists(_3_2_SPLIT_BY_MEANING):
        rprint(
            f"[yellow]⚠️ Files <{_3_1_SPLIT_BY_NLP}> and <{_3_2_SPLIT_BY_MEANING}> already exist, "
            "skip <split_by_spacy> step.[/yellow]"
        )
        return
    if os.path.exists(_3_1_SPLIT_BY_NLP) and not os.path.exists(_3_2_SPLIT_BY_MEANING):
        shutil.copyfile(_3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING)
        return

    language = get_source_language()
    rprint(
        f"[blue]🔍 Splitting sentences by LLM-restored punctuation (language: {language})...[/blue]"
    )
    chunks = pd.read_excel(_2_CLEANED_CHUNKS)
    words = chunks["text"].apply(lambda x: str(x).strip('"').strip("")).tolist()

    sentences = restore_and_split_words(words, language=language)

    os.makedirs(os.path.dirname(_3_1_SPLIT_BY_NLP), exist_ok=True)
    output_text = "\n".join(sentences) + ("\n" if sentences else "")
    with open(_3_1_SPLIT_BY_NLP, "w", encoding="utf-8") as f:
        f.write(output_text)

    with open(_3_2_SPLIT_BY_MEANING, "w", encoding="utf-8") as f:
        f.write(output_text)

    rprint(
        f"[green]💾 {len(sentences)} sentences saved to → `{_3_1_SPLIT_BY_NLP}` "
        f"and `{_3_2_SPLIT_BY_MEANING}`[/green]"
    )


if __name__ == "__main__":
    split_by_spacy()

from __future__ import annotations

import os
from typing import Sequence

import pandas as pd

from core.subtitle_gen.formatter import (
    convert_to_srt_format,
    format_subtitle_text,
    generate_subtitle_string,
    remove_punctuation,
)
from core.utils import log_event
from core.workspace import get_active_workspace


def show_difference(str1: str, str2: str) -> None:
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = [i for i in range(min_len) if str1[i] != str2[i]]
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))

    log_event(
        f"时间戳匹配差异: expected='{str1}' vs actual='{str2}' (pos={diff_positions})",
        tag="warn",
        label="SUBS",
        level="warn",
    )


def _levenshtein_fast(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein_fast(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _try_direct_utterance_timestamps(
    df_words: pd.DataFrame, sentences_list: list[str]
) -> list[tuple[float, float]] | None:
    """Read (start, end) in O(1) from workspace utterances word_span / sub_cues when matching."""
    ws = get_active_workspace()
    if not ws.has_utterances():
        return None
    try:
        utterances = ws.load_utterances()
    except Exception:
        return None
    if not utterances:
        return None

    n_words = len(df_words)
    all_spans_cover_df = int(utterances[-1].get("word_span", [0, 0])[1]) == n_words
    if [str(u.get("src", "")).strip() for u in utterances] == sentences_list:
        stamps: list[tuple[float, float]] = []
        for u in utterances:
            span = u.get("word_span")
            if not isinstance(span, list) or len(span) < 2:
                return None
            w_a, w_b = int(span[0]), int(span[1])
            if not (0 <= w_a < w_b <= n_words):
                return None
            if not all_spans_cover_df:
                w_a_text = remove_punctuation(
                    str(df_words["text"].iloc[w_a]).lower()
                ).replace(" ", "")
                u_clean = remove_punctuation(
                    str(u.get("src_raw") or u.get("src", "")).lower()
                ).replace(" ", "")
                has_span_overlap = any(
                    (
                        wt := remove_punctuation(
                            str(df_words["text"].iloc[k]).lower()
                        ).replace(" ", "")
                    )
                    and wt in u_clean
                    for k in range(w_a, w_b)
                )
                if (
                    w_a_text
                    and u_clean
                    and not u_clean.startswith(w_a_text[:1])
                    and not has_span_overlap
                ):
                    return None
            stamps.append(
                (
                    float(df_words["start"].iloc[w_a]),
                    float(df_words["end"].iloc[w_b - 1]),
                )
            )
        return stamps

    sub_cues = [
        c for u in utterances for c in (u.get("sub_cues") or []) if isinstance(c, dict)
    ]
    if sub_cues and [str(c.get("src", "")).strip() for c in sub_cues] == sentences_list:
        stamps = []
        for c in sub_cues:
            span = c.get("word_span")
            if not isinstance(span, list) or len(span) < 2:
                return None
            w_a, w_b = int(span[0]), int(span[1])
            if not (0 <= w_a < w_b <= n_words):
                return None
            if not all_spans_cover_df:
                w_a_text = remove_punctuation(
                    str(df_words["text"].iloc[w_a]).lower()
                ).replace(" ", "")
                c_clean = remove_punctuation(str(c.get("src", "")).lower()).replace(
                    " ", ""
                )
                has_span_overlap = any(
                    (
                        wt := remove_punctuation(
                            str(df_words["text"].iloc[k]).lower()
                        ).replace(" ", "")
                    )
                    and wt in c_clean
                    for k in range(w_a, w_b)
                )
                if (
                    w_a_text
                    and c_clean
                    and not c_clean.startswith(w_a_text[:1])
                    and not has_span_overlap
                ):
                    return None
            stamps.append(
                (
                    float(df_words["start"].iloc[w_a]),
                    float(df_words["end"].iloc[w_b - 1]),
                )
            )
        return stamps

    return None


def get_sentence_timestamps(
    df_words: pd.DataFrame, df_sentences: pd.DataFrame
) -> list[tuple[float, float]]:
    sentences_series = [
        "" if pd.isna(s) else str(s).strip() for s in df_sentences["Source"].tolist()
    ]
    direct_stamps = _try_direct_utterance_timestamps(df_words, sentences_series)
    if direct_stamps is not None:
        return direct_stamps

    raw_stamps: list[tuple[float, float] | None] = []

    full_words_str = ""
    position_to_word_idx: dict[int, int] = {}
    word_starts: list[int] = []
    word_ends: list[int] = []

    word_start_times = df_words["start"].tolist() if len(df_words) > 0 else []
    word_end_times = df_words["end"].tolist() if len(df_words) > 0 else []

    for idx, word in enumerate(df_words["text"]):
        word_str = "" if pd.isna(word) else str(word)
        clean_word = remove_punctuation(word_str.lower()).replace(" ", "")
        start_pos = len(full_words_str)
        full_words_str += clean_word
        end_pos = len(full_words_str)
        if end_pos > start_pos:
            word_starts.append(start_pos)
            word_ends.append(end_pos)
        for pos in range(start_pos, end_pos):
            position_to_word_idx[pos] = idx

    clean_sentences = [
        remove_punctuation(s.lower()).replace(" ", "") for s in sentences_series
    ]

    current_pos = 0
    for list_idx, (_, sentence) in enumerate(df_sentences["Source"].items()):
        clean_sentence = clean_sentences[list_idx]
        sentence_len = len(clean_sentence)

        if sentence_len == 0:
            raw_stamps.append(None)
            continue

        match_found = False
        pos = full_words_str.find(clean_sentence, current_pos)
        if pos != -1:
            start_word_idx = position_to_word_idx[pos]
            end_word_idx = position_to_word_idx[pos + sentence_len - 1]
            raw_stamps.append(
                (
                    float(word_start_times[start_word_idx]),
                    float(word_end_times[end_word_idx]),
                )
            )
            current_pos = pos + sentence_len
            match_found = True
        else:
            max_dist = min(
                max(1, sentence_len // 2),
                max(4, int(sentence_len * 0.35)),
            )
            win_low = max(0, current_pos - 8)
            win_high = min(len(full_words_str), current_pos + sentence_len + 16)
            cand_starts = [
                s
                for s in word_starts
                if win_low <= s <= min(win_high, current_pos + 12)
            ]
            cand_ends = [e for e in word_ends if win_low < e <= win_high]
            best_span: tuple[int, int] | None = None
            best_dist = max_dist + 1
            for s in cand_starts:
                for e in cand_ends:
                    if e <= s or abs((e - s) - sentence_len) > max_dist:
                        continue
                    d = _levenshtein_fast(full_words_str[s:e], clean_sentence)
                    if d < best_dist:
                        best_dist = d
                        best_span = (s, e)
            if best_span is not None and best_dist <= max_dist:
                s, e = best_span
                start_word_idx = position_to_word_idx[s]
                end_word_idx = position_to_word_idx[e - 1]
                raw_stamps.append(
                    (
                        float(word_start_times[start_word_idx]),
                        float(word_end_times[end_word_idx]),
                    )
                )
                current_pos = e
                match_found = True

        if not match_found:
            next_match_pos = -1
            for next_clean in clean_sentences[list_idx + 1 :]:
                if next_clean:
                    found_next = full_words_str.find(next_clean, current_pos)
                    if found_next != -1:
                        next_match_pos = found_next
                        break
            if next_match_pos != -1 or sentence_len <= 8:
                raw_stamps.append(None)
                if next_match_pos != -1:
                    current_pos = next_match_pos
                continue

            log_event(
                f"未找到句子精确时间戳匹配: {sentence}",
                tag="warn",
                label="SUBS",
                level="error",
            )
            show_difference(
                clean_sentence,
                full_words_str[current_pos : current_pos + len(clean_sentence)],
            )
            raise ValueError("❎ No match found for sentence.")

    last_word_end = float(word_end_times[-1]) if word_end_times else 0.0
    for i, stamp in enumerate(raw_stamps):
        if stamp is None:
            prev_stamp = raw_stamps[i - 1] if i > 0 else None
            start = prev_stamp[1] if prev_stamp is not None else 0.0
            following = next((s for s in raw_stamps[i + 1 :] if s is not None), None)
            end = (
                max(start, following[0])
                if following is not None
                else max(start, min(last_word_end, start))
            )
            raw_stamps[i] = (start, end)

    return [s for s in raw_stamps if s is not None]


def align_timestamp(
    df_text: pd.DataFrame,
    df_translate: pd.DataFrame,
    subtitle_output_configs: Sequence[tuple[str, list[str]]],
    output_dir: str | None,
    for_display: bool = True,
) -> pd.DataFrame:
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    stamps = list(get_sentence_timestamps(df_text, df_translate))
    for i in range(len(stamps) - 1):
        curr_start, curr_end = stamps[i]
        next_start, next_end = stamps[i + 1]
        delta_time = next_start - curr_end
        if 0 < delta_time <= 0.3:
            stamps[i] = (curr_start, next_start)
        elif delta_time < 0:
            midpoint = max(curr_start, min(next_end, curr_end + delta_time / 2))
            stamps[i] = (curr_start, midpoint)
            stamps[i + 1] = (midpoint, next_end)

    df_trans_time["timestamp"] = [convert_to_srt_format(s, e) for s, e in stamps]
    df_trans_time["duration"] = [e - s for s, e in stamps]

    if for_display and "Translation" in df_trans_time.columns:
        df_trans_time["Translation"] = df_trans_time["Translation"].apply(
            format_subtitle_text
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(subtitle_str)

    return df_trans_time

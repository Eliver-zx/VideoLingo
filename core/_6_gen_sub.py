import os
import re
import time
from typing import Any
import pandas as pd
from core.utils import get_joiner, load_key, load_key_or, log_event, report_progress  # noqa: F401
from core.utils.models import (
    _OUTPUT_DIR,
    resolve_task_path,
)
from core.workspace import get_active_workspace

SUBTITLE_OUTPUT_CONFIGS = [
    ("src.srt", ["Source"]),
    ("trans.srt", ["Translation"]),
    ("src_trans.srt", ["Source", "Translation"]),
    ("trans_src.srt", ["Translation", "Source"]),
]


def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""

    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"


def remove_punctuation(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []

    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)

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


_QUOTE_BOOK_MARK_TABLE = str.maketrans({c: None for c in "《》〈〉“”‘’「」『』"})


def _strip_quotes_and_dashes(text: str) -> str:
    s = str(text).translate(_QUOTE_BOOK_MARK_TABLE)
    if re.search(r"[\u4e00-\u9fff]", s):
        s = s.replace('"', "")
    s = re.sub(r"\s*(?:——|—|--)\s*", "，", s)
    s = re.sub(r"[：；]", "，", s)
    return s


def format_subtitle_text(x):
    """Format subtitle text: remove spaces between CJK and Latin/digits, strip quotes/dashes/colons, and replace CJK clause punctuation with spaces."""
    if pd.isna(x):
        return ""
    text = _strip_quotes_and_dashes(str(x))
    digit_pattern = r"[\d\uff10-\uff19]"
    text = re.sub(r"([\u4e00-\u9fff])\s+(" + digit_pattern + r")", r"\1\2", text)
    text = re.sub(r"(" + digit_pattern + r")\s+([\u4e00-\u9fff])", r"\1\2", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([a-zA-Z])", r"\1\2", text)
    text = re.sub(r"([a-zA-Z])\s+([\u4e00-\u9fff])", r"\1\2", text)
    text = re.sub(r"[，。、？！：；—]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _try_direct_utterance_timestamps(
    df_words, sentences_list: list[str]
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


def get_sentence_timestamps(df_words, df_sentences):
    sentences_series = [
        "" if pd.isna(s) else str(s).strip() for s in df_sentences["Source"].tolist()
    ]
    direct_stamps = _try_direct_utterance_timestamps(df_words, sentences_series)
    if direct_stamps is not None:
        return direct_stamps

    time_stamp_list = []

    # Build complete string and position mapping
    full_words_str = ""
    position_to_word_idx = {}
    word_starts = []
    word_ends = []

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

        # A line with nothing but punctuation has no words to match; place it after the loop
        if sentence_len == 0:
            time_stamp_list.append(None)
            continue

        match_found = False
        pos = full_words_str.find(clean_sentence, current_pos)
        if pos != -1:
            start_word_idx = position_to_word_idx[pos]
            end_word_idx = position_to_word_idx[pos + sentence_len - 1]
            time_stamp_list.append(
                (
                    float(df_words["start"][start_word_idx]),
                    float(df_words["end"][end_word_idx]),
                )
            )
            current_pos = pos + sentence_len
            match_found = True
        else:
            # Word-boundary fuzzy match fallback
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
            best_span = None
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
                time_stamp_list.append(
                    (
                        float(df_words["start"][start_word_idx]),
                        float(df_words["end"][end_word_idx]),
                    )
                )
                current_pos = e
                match_found = True

        if not match_found:
            # If a short phrase fails exact/fuzzy matching or a later sentence matches,
            # clamp to the interpolated span between the previous matched word and next sentence's start.
            next_match_pos = -1
            for next_clean in clean_sentences[list_idx + 1 :]:
                if next_clean:
                    found_next = full_words_str.find(next_clean, current_pos)
                    if found_next != -1:
                        next_match_pos = found_next
                        break
            if next_match_pos != -1 or sentence_len <= 8:
                time_stamp_list.append(None)
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

    # Punctuation-only or clamped short lines fill the gap between their neighbours
    last_word_end = float(df_words["end"].iloc[-1]) if len(df_words) > 0 else 0.0
    for i, stamp in enumerate(time_stamp_list):
        if stamp is None:
            start = (
                time_stamp_list[i - 1][1]
                if i > 0 and time_stamp_list[i - 1] is not None
                else 0.0
            )
            following = next(
                (s for s in time_stamp_list[i + 1 :] if s is not None), None
            )
            end = (
                max(start, following[0])
                if following
                else max(start, min(last_word_end, start))
            )
            time_stamp_list[i] = (start, end)

    return time_stamp_list


def align_timestamp(
    df_text,
    df_translate,
    subtitle_output_configs: list,
    output_dir: str | None,
    for_display: bool = True,
):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time["timestamp"] = time_stamp_list

    # Remove gaps and fix overlaps 🕳️
    for i in range(len(df_trans_time) - 1):
        curr_start, curr_end = df_trans_time.loc[i, "timestamp"]
        next_start, next_end = df_trans_time.loc[i + 1, "timestamp"]
        delta_time = next_start - curr_end
        if 0 < delta_time <= 0.3:
            df_trans_time.at[i, "timestamp"] = (curr_start, next_start)
        elif delta_time < 0:
            midpoint = max(curr_start, min(next_end, curr_end + delta_time / 2))
            df_trans_time.at[i, "timestamp"] = (curr_start, midpoint)
            df_trans_time.at[i + 1, "timestamp"] = (midpoint, next_end)

    df_trans_time["duration"] = df_trans_time["timestamp"].apply(lambda x: x[1] - x[0])

    # Convert start and end timestamps to SRT format
    df_trans_time["timestamp"] = df_trans_time["timestamp"].apply(
        lambda x: convert_to_srt_format(x[0], x[1])
    )

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display and "Translation" in df_trans_time.columns:
        df_trans_time["Translation"] = df_trans_time["Translation"].apply(
            format_subtitle_text
        )

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return "".join(
            [
                f"{i + 1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n"
                for i, row in df.iterrows()
            ]
        ).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(subtitle_str)

    return df_trans_time


def gen_source_subtitles():
    """Transcription only: `src.srt` from the split sentences, without a translation."""
    from core._1_ytdlp import find_subtitle_file
    from core._5_split_sub import split_source_lines

    out_dir = resolve_task_path(_OUTPUT_DIR)
    ws = get_active_workspace()

    words = ws.load_words()
    df_text = (
        pd.DataFrame(words) if words else pd.DataFrame(columns=["text", "start", "end"])
    )

    utts = ws.load_utterances()
    lines = [
        str(u.get("src") or "").strip() for u in utts if str(u.get("src") or "").strip()
    ]

    # The subtitles of the user keep their lines
    df_source = pd.DataFrame(
        {"Source": lines if find_subtitle_file() else split_source_lines(lines)}
    )
    align_timestamp(
        df_text, df_source, [("src.srt", ["Source"])], out_dir, for_display=False
    )
    src_file = os.path.join(out_dir, "src.srt")
    if os.path.isfile(src_file):
        ws.set_kv("srt_synced_mtime", os.path.getmtime(src_file))
    ws.notify_artifact_update(
        "subtitles",
        message=f"已生成原文 SRT 字幕 ({len(df_source)} 条)",
        tag="llm",
        label="SUBS",
    )
    report_progress(
        1.0,
        f"已生成 {len(df_source)} 条原文字幕",
        f"原文 SRT 字幕生成完成 ({out_dir}/src.srt)",
        tag="llm",
        label="SUBS",
    )


# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ""
    return _strip_quotes_and_dashes(str(x)).strip("。，、？！：；—").strip()


def align_timestamp_main():
    out_dir = resolve_task_path(_OUTPUT_DIR)
    ws = get_active_workspace()

    words = ws.load_words()
    df_text = (
        pd.DataFrame(words) if words else pd.DataFrame(columns=["text", "start", "end"])
    )

    utts = ws.load_utterances()
    cue_rows = []
    for u in utts:
        cues = u.get("sub_cues")
        if isinstance(cues, list) and cues:
            for c in cues:
                cue_rows.append(
                    {
                        "Source": str(c.get("src") or ""),
                        "Translation": str(
                            c.get("tgt", c.get("trans", c.get("text", ""))) or ""
                        ),
                    }
                )
        else:
            cue_rows.append(
                {
                    "Source": str(u.get("src") or ""),
                    "Translation": str(u.get("tgt") or ""),
                }
            )
    df_translate = pd.DataFrame(cue_rows, columns=["Source", "Translation"])
    df_translate["Translation"] = df_translate["Translation"].apply(clean_translation)

    align_timestamp(
        df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, out_dir, for_display=True
    )
    srt_mtimes = [
        os.path.getmtime(os.path.join(out_dir, fn))
        for fn in ("src.srt", "trans.srt")
        if os.path.isfile(os.path.join(out_dir, fn))
    ]
    if srt_mtimes:
        ws.set_kv("srt_synced_mtime", max(srt_mtimes))

    ws.notify_artifact_update(
        "subtitles",
        message=f"已生成双语 SRT 字幕文件 ({len(df_translate)} 条)",
        tag="llm",
        label="SUBS",
    )
    report_progress(
        1.0,
        f"已生成 {len(df_translate)} 条双语字幕",
        f"双语字幕已写入 ({out_dir})",
        tag="llm",
        label="SUBS",
    )


_SRT_TIMECODE_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)
_SRT_TAGS_RE = re.compile(r"<[^>]*>|\{\\[^}]*\}")


def _parse_srt_blocks(path: str | os.PathLike) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        raw_bytes = f.read()
    try:
        text = raw_bytes.decode(
            "utf-16"
            if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff"))
            else "utf-8-sig"
        )
    except UnicodeDecodeError:
        text = raw_bytes.decode("utf-8", errors="replace")

    blocks: list[tuple[int, float, float, list[str]]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _SRT_TIMECODE_RE.search(line)
        if m:
            cue_num = len(blocks) + 1
            if i > 0 and lines[i - 1].strip().isdigit():
                cue_num = int(lines[i - 1].strip())
                if blocks:
                    prev_blank = i >= 2 and not lines[i - 2].strip()
                    non_blank_prev = [ln for ln in blocks[-1][3] if ln.strip()]
                    if prev_blank or (
                        len(non_blank_prev) > 1 and cue_num <= len(blocks) + 2
                    ):
                        blocks[-1][3].pop()
            h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
            start = round(
                int(h1) * 3600 + int(m1) * 60 + float(f"{s1}.{ms1.ljust(3, '0')[:3]}"),
                3,
            )
            end = round(
                int(h2) * 3600 + int(m2) * 60 + float(f"{s2}.{ms2.ljust(3, '0')[:3]}"),
                3,
            )
            blocks.append((cue_num, start, end, []))
        elif blocks:
            blocks[-1][3].append(line)

    parsed: list[dict[str, Any]] = []
    for num, start, end, raw_lines in blocks:
        cleaned = " ".join(
            _SRT_TAGS_RE.sub("", ln).strip() for ln in raw_lines if ln.strip()
        ).strip()
        parsed.append(
            {
                "number": num,
                "start": start,
                "end": end,
                "timestamp": convert_to_srt_format(start, end),
                "text": cleaned,
            }
        )
    return parsed


def sync_subtitles_from_srt() -> dict[str, int]:
    """Sync edited `src.srt` and `trans.srt` from disk back into `project.sqlite` and bilingual SRT files."""
    out_dir = resolve_task_path(_OUTPUT_DIR)
    ws = get_active_workspace()
    src_path = os.path.join(out_dir, "src.srt")
    trans_path = os.path.join(out_dir, "trans.srt")

    has_src = os.path.isfile(src_path)
    has_trans = os.path.isfile(trans_path)
    if not has_src and not has_trans:
        raise FileNotFoundError("未找到 src.srt 或 trans.srt，无法同步字幕。")

    src_blocks = _parse_srt_blocks(src_path) if has_src else []
    trans_blocks = _parse_srt_blocks(trans_path) if has_trans else []
    if not src_blocks and not trans_blocks:
        raise ValueError("SRT 字幕文件为空或格式无效。")

    src_mtime = os.path.getmtime(src_path) if has_src else 0.0
    trans_mtime = os.path.getmtime(trans_path) if has_trans else 0.0
    prefer_src_ts = has_src and (not has_trans or src_mtime > trans_mtime + 1e-3)

    merged_cues: list[dict[str, Any]] = []
    if src_blocks and trans_blocks:
        if len(src_blocks) == len(trans_blocks):
            for sb, tb in zip(src_blocks, trans_blocks):
                ts_block = sb if prefer_src_ts else tb
                merged_cues.append(
                    {
                        "src": sb["text"],
                        "tgt": tb["text"],
                        "start": ts_block["start"],
                        "end": ts_block["end"],
                        "timestamp": ts_block["timestamp"],
                    }
                )
        else:
            src_by_num = {sb["number"]: sb for sb in src_blocks}
            for idx, tb in enumerate(trans_blocks):
                sb = src_by_num.get(tb["number"])
                if sb is None and idx < len(src_blocks):
                    sb = src_blocks[idx]
                merged_cues.append(
                    {
                        "src": sb["text"] if sb else "",
                        "tgt": tb["text"],
                        "start": tb["start"],
                        "end": tb["end"],
                        "timestamp": tb["timestamp"],
                    }
                )
    elif src_blocks:
        for sb in src_blocks:
            merged_cues.append(
                {
                    "src": sb["text"],
                    "tgt": "",
                    "start": sb["start"],
                    "end": sb["end"],
                    "timestamp": sb["timestamp"],
                }
            )
    else:
        for tb in trans_blocks:
            merged_cues.append(
                {
                    "src": "",
                    "tgt": tb["text"],
                    "start": tb["start"],
                    "end": tb["end"],
                    "timestamp": tb["timestamp"],
                }
            )

    utts = ws.load_utterances()
    existing_cue_total = sum(
        max(
            1,
            len(u.get("sub_cues") or []) if isinstance(u.get("sub_cues"), list) else 1,
        )
        for u in utts
    )

    if utts and existing_cue_total == len(merged_cues):
        cursor = 0
        for u in utts:
            raw_cues = u.get("sub_cues")
            old_cues = (
                raw_cues
                if isinstance(raw_cues, list) and raw_cues
                else [
                    {
                        "src": u.get("src", ""),
                        "tgt": u.get("tgt", ""),
                        "word_span": list(u.get("word_span") or [0, 0]),
                        "start": float(u.get("start") or 0.0),
                        "end": float(u.get("end") or 0.0),
                    }
                ]
            )
            new_cues: list[dict[str, Any]] = []
            src_changed = False
            tgt_changed = False
            for oc in old_cues:
                mc = merged_cues[cursor]
                cursor += 1
                old_s = str(oc.get("src") or "").strip()
                old_t = str(oc.get("tgt", oc.get("trans", "")) or "").strip()
                if mc["src"] != old_s:
                    src_changed = True
                if mc["tgt"] != old_t and mc["tgt"] != format_subtitle_text(old_t):
                    tgt_changed = True
                new_cues.append(
                    {
                        **oc,
                        "src": mc["src"],
                        "tgt": mc["tgt"],
                        "trans": mc["tgt"],
                        "start": mc["start"],
                        "end": mc["end"],
                    }
                )
            u["sub_cues"] = new_cues
            u["start"] = new_cues[0]["start"]
            u["end"] = new_cues[-1]["end"]
            if src_changed or not str(u.get("src") or "").strip():
                u["src"] = " ".join(c["src"] for c in new_cues if c["src"]).strip()
                u["src_punct"] = u["src"]
            if tgt_changed or not str(u.get("tgt") or "").strip():
                u["tgt"] = " ".join(c["tgt"] for c in new_cues if c["tgt"]).strip()
                u["tts_text"] = u["tgt"]
        utts_out = utts
    elif utts:
        buckets: list[list[dict[str, Any]]] = [[] for _ in utts]
        for mc in merged_cues:
            mid = (mc["start"] + mc["end"]) / 2.0
            best_idx = min(
                range(len(utts)),
                key=lambda i: (
                    0.0
                    if float(utts[i].get("start", 0.0))
                    <= mid
                    <= float(utts[i].get("end", 0.0))
                    else min(
                        abs(mid - float(utts[i].get("start", 0.0))),
                        abs(mid - float(utts[i].get("end", 0.0))),
                    )
                ),
            )
            buckets[best_idx].append(mc)
        utts_out = []
        for u, b_cues in zip(utts, buckets):
            if not b_cues:
                continue
            item = dict(u)
            w_span = list(item.get("word_span") or [0, 0])
            new_cues = [
                {
                    "src": mc["src"],
                    "tgt": mc["tgt"],
                    "trans": mc["tgt"],
                    "word_span": w_span,
                    "start": mc["start"],
                    "end": mc["end"],
                }
                for mc in b_cues
            ]
            item["sub_cues"] = new_cues
            item["start"] = new_cues[0]["start"]
            item["end"] = new_cues[-1]["end"]
            item["src"] = " ".join(c["src"] for c in new_cues if c["src"]).strip()
            item["src_punct"] = item["src"]
            item["tgt"] = " ".join(c["tgt"] for c in new_cues if c["tgt"]).strip()
            item["tts_text"] = item["tgt"]
            utts_out.append(item)
    else:
        utts_out = [
            {
                "id": f"u_{i:04d}",
                "index": i,
                "start": mc["start"],
                "end": mc["end"],
                "gap_after": 0.0,
                "word_span": [0, 0],
                "src": mc["src"],
                "src_punct": mc["src"],
                "tgt": mc["tgt"],
                "tts_text": mc["tgt"],
                "sub_cues": [
                    {
                        "src": mc["src"],
                        "tgt": mc["tgt"],
                        "trans": mc["tgt"],
                        "word_span": [0, 0],
                        "start": mc["start"],
                        "end": mc["end"],
                    }
                ],
            }
            for i, mc in enumerate(merged_cues)
        ]

    sync_ts = max(time.time(), src_mtime, trans_mtime)
    rev = ws.save_utterances(utts_out, mtime=sync_ts)
    ws.set_kv("sub_split_rev", rev, mtime=sync_ts)
    ws.set_kv("srt_synced_mtime", max(src_mtime, trans_mtime), mtime=sync_ts)
    if ws.has_dub_tasks():
        ws.invalidate_from_stage("dub_tts")

    if trans_blocks:
        src_trans_str = "".join(
            f"{i + 1}\n{mc['timestamp']}\n{mc['src']}\n{mc['tgt']}\n\n"
            for i, mc in enumerate(merged_cues)
        ).strip()
        trans_src_str = "".join(
            f"{i + 1}\n{mc['timestamp']}\n{mc['tgt']}\n{mc['src']}\n\n"
            for i, mc in enumerate(merged_cues)
        ).strip()
        with open(os.path.join(out_dir, "src_trans.srt"), "w", encoding="utf-8") as f:
            f.write(src_trans_str)
        with open(os.path.join(out_dir, "trans_src.srt"), "w", encoding="utf-8") as f:
            f.write(trans_src_str)

    ws.notify_artifact_update(
        "subtitles",
        message=f"已从 SRT 同步更新 {len(merged_cues)} 轴字幕",
        tag="llm",
        label="SUBS",
    )
    return {"cues": len(merged_cues), "utterances": len(utts_out)}


def sync_subtitles_from_srt_if_newer() -> bool:
    """Auto-sync `src.srt` / `trans.srt` into SQLite if modified on disk after the last DB/SRT save."""
    out_dir = resolve_task_path(_OUTPUT_DIR)
    ws = get_active_workspace()
    src_path = os.path.join(out_dir, "src.srt")
    trans_path = os.path.join(out_dir, "trans.srt")
    mtimes = [os.path.getmtime(p) for p in (src_path, trans_path) if os.path.isfile(p)]
    if not mtimes:
        return False
    baseline_mtime = max(
        ws.get_utterances_mtime(),
        float(ws.get_kv("srt_synced_mtime", 0.0) or 0.0),
    )
    if max(mtimes) > baseline_mtime + 1e-3:
        sync_subtitles_from_srt()
        return True
    return False


if __name__ == "__main__":
    align_timestamp_main()

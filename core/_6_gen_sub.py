import os
import re
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.utils import get_joiner, load_key, load_key_or, rprint  # noqa: F401
from core.utils.models import (
    _2_CLEANED_CHUNKS,
    _3_2_SPLIT_BY_MEANING,
    _5_REMERGED,
    _5_SPLIT_SUB,
    _AUDIO_DIR,
    _OUTPUT_DIR,
    resolve_task_path,
)
from core.utils.subtitle_style import build_ass_from_srt_pair, get_burn_mode
from core.workspace import get_active_workspace

console = Console()

SUBTITLE_OUTPUT_CONFIGS = [
    ("src.srt", ["Source"]),
    ("trans.srt", ["Translation"]),
    ("src_trans.srt", ["Source", "Translation"]),
    ("trans_src.srt", ["Translation", "Source"]),
    ("bi.srt", ["Translation", "Source"]),
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ("src_subs_for_audio.srt", ["Source"]),
    ("trans_subs_for_audio.srt", ["Translation"]),
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

    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print(
        "Position markers: "
        + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2))))
    )
    print(f"Difference indices: {diff_positions}")


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


def format_subtitle_text(x):
    """Format subtitle text: remove spaces between CJK and Latin/digits, replace CJK clause punctuation with spaces."""
    if pd.isna(x):
        return ""
    text = str(x)
    digit_pattern = r"[\d\uff10-\uff19]"
    text = re.sub(r"([\u4e00-\u9fff])\s+(" + digit_pattern + r")", r"\1\2", text)
    text = re.sub(r"(" + digit_pattern + r")\s+([\u4e00-\u9fff])", r"\1\2", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([a-zA-Z])", r"\1\2", text)
    text = re.sub(r"([a-zA-Z])\s+([\u4e00-\u9fff])", r"\1\2", text)
    text = re.sub(r"[，。、？！：；]", " ", text)
    return text.strip()


def _try_direct_utterance_timestamps(df_words, sentences_list: list[str]) -> list[tuple[float, float]] | None:
    """Read (start, end) in O(1) from utterances.json word_span / sub_cues when matching."""
    ws = get_active_workspace()
    if not ws.utterances_path.is_file():
        return None
    try:
        utterances = ws.load_utterances()
    except Exception:
        return None
    if not utterances:
        return None

    n_words = len(df_words)
    if [str(u.get("src", "")).strip() for u in utterances] == sentences_list:
        stamps: list[tuple[float, float]] = []
        for u in utterances:
            span = u.get("word_span")
            if not isinstance(span, list) or len(span) < 2:
                return None
            w_a, w_b = int(span[0]), int(span[1])
            if not (0 <= w_a < w_b <= n_words):
                return None
            w_a_Text = remove_punctuation(str(df_words["text"].iloc[w_a]).lower()).replace(" ", "")
            u_clean = remove_punctuation(str(u.get("src_raw") or u.get("src", "")).lower()).replace(" ", "")
            if w_a_Text and u_clean and not u_clean.startswith(w_a_Text[:1]):
                return None
            stamps.append((float(df_words["start"].iloc[w_a]), float(df_words["end"].iloc[w_b - 1])))
        return stamps

    sub_cues = [c for u in utterances for c in (u.get("sub_cues") or []) if isinstance(c, dict)]
    if sub_cues and [str(c.get("src", "")).strip() for c in sub_cues] == sentences_list:
        stamps = []
        for c in sub_cues:
            span = c.get("word_span")
            if not isinstance(span, list) or len(span) < 2:
                return None
            w_a, w_b = int(span[0]), int(span[1])
            if not (0 <= w_a < w_b <= n_words):
                return None
            w_a_text = remove_punctuation(str(df_words["text"].iloc[w_a]).lower()).replace(" ", "")
            c_clean = remove_punctuation(str(c.get("src", "")).lower()).replace(" ", "")
            if w_a_text and c_clean and not c_clean.startswith(w_a_text[:1]):
                # Also check parent utterance src_raw in case fix_source or user edit changed the first word
                if int(utterances[-1].get("word_span", [0, 0])[1]) != n_words:
                    return None
            stamps.append((float(df_words["start"].iloc[w_a]), float(df_words["end"].iloc[w_b - 1])))
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
            max_dist = max(2, int(sentence_len * 0.25))
            win_low = max(0, current_pos - 8)
            win_high = min(len(full_words_str), current_pos + sentence_len + 16)
            cand_starts = [s for s in word_starts if win_low <= s <= min(win_high, current_pos + 12)]
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

            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(
                clean_sentence,
                full_words_str[current_pos : current_pos + len(clean_sentence)],
            )
            print("\nOriginal sentence:", sentence)
            raise ValueError("❎ No match found for sentence.")

    # Punctuation-only or clamped short lines fill the gap between their neighbours
    last_word_end = float(df_words["end"].iloc[-1]) if len(df_words) > 0 else 0.0
    for i, stamp in enumerate(time_stamp_list):
        if stamp is None:
            start = time_stamp_list[i - 1][1] if i > 0 and time_stamp_list[i - 1] is not None else 0.0
            following = next((s for s in time_stamp_list[i + 1 :] if s is not None), None)
            end = max(start, following[0]) if following else max(start, min(last_word_end, start))
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
        if 0 < delta_time < 1:
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
        df_trans_time["Translation"] = df_trans_time["Translation"].apply(format_subtitle_text)

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return "".join(
            [
                f"{i+1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n"
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

    cleaned_path = resolve_task_path(_2_CLEANED_CHUNKS)
    meaning_path = resolve_task_path(_3_2_SPLIT_BY_MEANING)
    out_dir = resolve_task_path(_OUTPUT_DIR)

    df_text = pd.read_excel(cleaned_path)
    df_text["text"] = df_text["text"].str.strip('"').str.strip()
    with open(meaning_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # The subtitles of the user keep their lines
    df_source = pd.DataFrame(
        {"Source": lines if find_subtitle_file() else split_source_lines(lines)}
    )
    align_timestamp(df_text, df_source, [("src.srt", ["Source"])], out_dir, for_display=False)
    console.print(Panel(f"[bold green]🎉📝 Source subtitles are ready: `{out_dir}/src.srt`[/bold green]"))


# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ""
    return str(x).strip("。，、？！：；").strip()


def align_timestamp_main():
    cleaned_path = resolve_task_path(_2_CLEANED_CHUNKS)
    split_sub_path = resolve_task_path(_5_SPLIT_SUB)
    remerged_path = resolve_task_path(_5_REMERGED)
    out_dir = resolve_task_path(_OUTPUT_DIR)
    audio_dir = resolve_task_path(_AUDIO_DIR)

    df_text = pd.read_excel(cleaned_path)
    df_text["text"] = df_text["text"].str.strip('"').str.strip()
    df_translate = pd.read_excel(split_sub_path)
    df_translate["Translation"] = df_translate["Translation"].apply(clean_translation)

    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, out_dir, for_display=True)

    ws = get_active_workspace()
    manifest_media = (ws.load_manifest().get("media") or {}) if ws.manifest_path.is_file() else {}
    disp_w = int(manifest_media.get("display_width") or 1920)
    disp_h = int(manifest_media.get("display_height") or 1080)
    build_ass_from_srt_pair(
        os.path.join(out_dir, "src.srt"),
        os.path.join(out_dir, "trans.srt"),
        os.path.join(out_dir, "bi.ass"),
        mode=get_burn_mode("subtitle"),
        display_width=disp_w,
        display_height=disp_h,
    )
    console.print(
        Panel(
            f"[bold green]🎉📝 Subtitles generation completed! Please check in the `{out_dir}` folder 👀[/bold green]"
        )
    )

    # for audio: keep full punctuation (for_display=False) so TTS preserves natural prosody pauses
    df_translate_for_audio = pd.read_excel(remerged_path)
    df_translate_for_audio["Translation"] = (
        df_translate_for_audio["Translation"].fillna("").astype(str).str.strip()
    )
    if ws.utterances_path.is_file():
        utts = ws.load_utterances()
        if len(utts) == len(df_translate_for_audio):
            for idx, u in enumerate(utts):
                tts_val = str(u.get("tts_text") or "").strip()
                tgt_val = str(u.get("tgt") or "").strip()
                if tts_val and tts_val != tgt_val:
                    df_translate_for_audio.at[idx, "Translation"] = tts_val

    align_timestamp(
        df_text,
        df_translate_for_audio,
        AUDIO_SUBTITLE_OUTPUT_CONFIGS,
        audio_dir,
        for_display=False,
    )
    console.print(
        Panel(
            f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{audio_dir}` folder 👀[/bold green]"
        )
    )


if __name__ == "__main__":
    align_timestamp_main()

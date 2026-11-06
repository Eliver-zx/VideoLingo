import datetime
import re
from core._8_1_audio_task import time_diff_seconds
from core.asr_backend.audio_preprocess import get_audio_duration
from core.tts_backend.estimate_duration import init_estimator, estimate_duration
from core.utils import load_key, log_event, report_progress
from core.utils.models import (
    _RAW_AUDIO_FILE,
    resolve_task_path,
)
from core.workspace import get_active_workspace

SRC_SRT = "output/src.srt"
TRANS_SRT = "output/trans.srt"
MAX_MERGE_COUNT = 5
ESTIMATOR = None


def clean_line(text):
    """A line of the subtitles without the remarks in brackets and without the dashes of a dialogue."""
    text = re.sub(r"\([^)]*\)|（[^）]*）", "", text)
    text = re.sub(r"\([^)]*$|（[^）]*$", "", text)
    text = re.sub(r"^[^(]*\)|^[^（]*）", "", text)
    # A hyphen inside a word is a part of the word: "hands-on"
    return re.sub(r"(?<!\w)-|-(?!\w)", "", text).strip()


def calc_if_too_fast(est_dur, tol_dur, duration, tolerance):
    accept = load_key("speed_factor.accept")  # Maximum acceptable speed factor
    if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
        return 2
    elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
        return 1
    elif est_dur < duration - tolerance:  # Speaking speed too slow
        return -1
    else:  # Normal speaking speed
        return 0


def merge_rows(df, start_idx, merge_count):
    """Merge multiple rows and calculate cumulative values"""
    merged = {
        "est_dur": df.iloc[start_idx]["est_dur"],
        "tol_dur": df.iloc[start_idx]["tol_dur"],
        "duration": df.iloc[start_idx]["duration"],
    }

    while merge_count < MAX_MERGE_COUNT and (start_idx + merge_count) < len(df):
        if df.iloc[start_idx + merge_count - 1]["cut_off"] == 1:
            break
        next_row = df.iloc[start_idx + merge_count]
        merged["est_dur"] += next_row["est_dur"]
        merged["tol_dur"] += next_row["tol_dur"]
        merged["duration"] += next_row["duration"]

        speed_flag = calc_if_too_fast(
            merged["est_dur"],
            merged["tol_dur"],
            merged["duration"],
            df.iloc[start_idx + merge_count]["tolerance"],
        )

        if speed_flag <= 0 or merge_count == 2:
            df.at[start_idx + merge_count, "cut_off"] = 1
            return merge_count + 1

        merge_count += 1

    # If no suitable merge point is found or a cut_off boundary was hit
    df.at[start_idx + merge_count - 1, "cut_off"] = 1
    return merge_count


def analyze_subtitle_timing_and_speed(df):
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    estimator = ESTIMATOR
    TOLERANCE = load_key("tolerance")
    whole_dur = get_audio_duration(resolve_task_path(_RAW_AUDIO_FILE))
    df["gap"] = 0.0  # Initialize gap column
    for i in range(len(df) - 1):
        current_end = datetime.datetime.strptime(
            df.loc[i, "end_time"], "%H:%M:%S.%f"
        ).time()
        next_start = datetime.datetime.strptime(
            df.loc[i + 1, "start_time"], "%H:%M:%S.%f"
        ).time()
        df.loc[i, "gap"] = max(
            0.0, time_diff_seconds(current_end, next_start, datetime.date.today())
        )

    # Set the gap for the last line
    last_end = datetime.datetime.strptime(df.iloc[-1]["end_time"], "%H:%M:%S.%f").time()
    last_end_seconds = (
        last_end.hour * 3600
        + last_end.minute * 60
        + last_end.second
        + last_end.microsecond / 1000000
    )
    df.iloc[-1, df.columns.get_loc("gap")] = max(0.0, whole_dur - last_end_seconds)

    df["tolerance"] = df["gap"].apply(lambda x: max(0.0, min(TOLERANCE, x)))
    df["tol_dur"] = df["duration"] + df["tolerance"]
    df["est_dur"] = df.apply(lambda x: estimate_duration(x["text"], estimator), axis=1)

    ## Calculate speed indicators
    accept = load_key("speed_factor.accept")  # Maximum acceptable speed factor

    def calc_if_too_fast(row):
        est_dur = row["est_dur"]
        tol_dur = row["tol_dur"]
        duration = row["duration"]
        tolerance = row["tolerance"]

        if est_dur / accept > tol_dur:  # Even max speed factor cannot adapt
            return 2
        elif est_dur > tol_dur:  # Speed adjustment needed within acceptable range
            return 1
        elif est_dur < duration - tolerance:  # Speaking speed too slow
            return -1
        else:  # Normal speaking speed
            return 0

    df["if_too_fast"] = df.apply(calc_if_too_fast, axis=1)
    return df


def process_cutoffs(df):
    df["cut_off"] = 0  # Initialize cut_off column
    df.loc[df["gap"] >= load_key("tolerance"), "cut_off"] = (
        1  # Set to 1 when gap is greater than TOLERANCE
    )
    idx = 0
    while idx < len(df):
        # Process marked split points
        if df.iloc[idx]["cut_off"] == 1:
            if df.iloc[idx]["if_too_fast"] == 2:
                log_event(
                    f"第 {idx + 1} 条配音语速过快，超出最大倍速阈值",
                    tag="warn",
                    label="TTS",
                    level="warn",
                )
            idx += 1
            continue

        # Process the last line
        if idx + 1 >= len(df):
            df.at[idx, "cut_off"] = 1
            break

        # Process normal or slow lines
        if df.iloc[idx]["if_too_fast"] <= 0:
            if df.iloc[idx + 1]["if_too_fast"] <= 0:
                df.at[idx, "cut_off"] = 1
                idx += 1
            else:
                idx += merge_rows(df, idx, 1)
        # Process fast lines
        else:
            idx += merge_rows(df, idx, 1)

    return df


def gen_dub_chunks():
    ws = get_active_workspace()
    if ws.is_step_done("dub_tasks"):
        report_progress(
            1.0,
            "复用已有配音分块",
            "配音分块已是最新，跳过重复计算",
            tag="asr",
            label="CACHE",
        )
        return
    report_progress(
        0.60,
        "分析配音语速与停顿…",
        "正在分析配音语速与停顿切点",
        tag="asr",
        label="TTS",
    )
    df = ws.load_dub_tasks_df()

    df = analyze_subtitle_timing_and_speed(df)

    df = process_cutoffs(df)

    content_lines = []
    ori_content_lines = []
    trans_srt_path = resolve_task_path(TRANS_SRT)
    src_srt_path = resolve_task_path(SRC_SRT)
    with open(trans_srt_path, "r", encoding="utf-8") as f_trans:
        content = f_trans.read()
    with open(src_srt_path, "r", encoding="utf-8") as f_src:
        ori_content = f_src.read()

    for block in content.strip().split("\n\n"):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 3:
            text = " ".join(lines[2:])
            text = clean_line(text)
            content_lines.append(text)

    for block in ori_content.strip().split("\n\n"):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 3:
            text = " ".join(lines[2:])
            text = clean_line(text)
            ori_content_lines.append(text)

    # Match processing
    df["lines"] = None
    df["src_lines"] = None
    last_idx = 0

    def clean_text(text):
        """clean space and punctuation, stripping parenthetical remarks including split ones"""
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r"\([^)]*\)|（[^）]*）", "", text)
        text = re.sub(r"[()（）]", "", text)
        return re.sub(r"[^\w\s]|[\s]", "", text)

    for pos_idx in range(len(df)):
        row = df.iloc[pos_idx]
        target = clean_text(row["text"])
        matches = []
        current = ""
        match_indices = []  # Store indices for matching lines

        for i in range(last_idx, len(content_lines)):
            line = content_lines[i]
            cleaned_line = clean_text(line)
            if not cleaned_line:
                continue
            current += cleaned_line
            matches.append(line)  # 存储原始文本
            match_indices.append(i)

            if current == target:
                df.at[pos_idx, "lines"] = matches
                df.at[pos_idx, "src_lines"] = [
                    ori_content_lines[j] if j < len(ori_content_lines) else ""
                    for j in match_indices
                ]
                last_idx = i + 1
                break
        else:
            log_event(
                f"第 {pos_idx + 1} 条配音分块字幕匹配失败: target='{target[:40]}'",
                tag="warn",
                label="TTS",
                level="error",
            )
            raise ValueError("Matching failed")

    ws.save_dub_tasks(df)
    report_progress(
        1.0,
        f"已完成 {len(df)} 条配音分块规划",
        f"配音语速与分块规划完成（共 {len(df)} 条）",
        tag="asr",
        label="TTS",
    )


if __name__ == "__main__":
    gen_dub_chunks()

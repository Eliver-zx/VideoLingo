import datetime
import re
import pandas as pd
from core.utils import check_file_exists, load_key, log_event, report_progress
from core.utils.models import resolve_task_path
from core.workspace import get_active_workspace

TRANS_SRT = "output/trans.srt"
SRC_SRT = "output/src.srt"


def time_diff_seconds(t1, t2, base_date):
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()


def _clean_spoken_text(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(text or "")).strip()
    text = re.sub(r"（[^）]*）", "", text).strip()
    text = re.sub(r"\([^)]*$|（[^）]*$", "", text).strip()
    text = re.sub(r"^[^(]*\)|^[^（]*）", "", text).strip()
    return re.sub(r"(?<!\w)-|-(?!\w)", "", text).strip()


def process_srt():
    """Build dubbing tasks from `src.srt` and `trans.srt`."""
    trans_subs_path = resolve_task_path(TRANS_SRT)
    src_subs_path = resolve_task_path(SRC_SRT)
    subtitles = []

    with open(trans_subs_path, "r", encoding="utf-8") as file:
        content = file.read()

    with open(src_subs_path, "r", encoding="utf-8") as src_file:
        src_content = src_file.read()

    src_subtitles = {}

    for block in src_content.strip().split("\n\n"):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue

        number = int(lines[0])
        src_text = " ".join(lines[2:])
        src_subtitles[number] = src_text

    for block in content.strip().split("\n\n"):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue

        try:
            number = int(lines[0])
            start_time, end_time = lines[1].split(" --> ")
            start_time = datetime.datetime.strptime(start_time, "%H:%M:%S,%f").time()
            end_time = datetime.datetime.strptime(end_time, "%H:%M:%S,%f").time()
            duration = time_diff_seconds(start_time, end_time, datetime.date.today())
            text = _clean_spoken_text(" ".join(lines[2:]))
            if not text:
                continue
            origin = src_subtitles.get(number, "")

        except ValueError as e:
            log_event(
                f"无法解析字幕块 '{block[:40]}': {e}，已跳过",
                tag="warn",
                label="TTS",
                level="warn",
            )
            continue

        subtitles.append(
            {
                "number": number,
                "start_time": start_time,
                "end_time": end_time,
                "raw_start_time": start_time,
                "raw_end_time": end_time,
                "duration": duration,
                "text": text,
                "origin": origin,
            }
        )

    df = pd.DataFrame(subtitles)
    if df.empty:
        return df

    i = 0
    MIN_SUB_DUR = load_key("min_subtitle_duration")
    while i < len(df):
        today = datetime.date.today()
        if df.loc[i, "duration"] < MIN_SUB_DUR:
            if (
                i < len(df) - 1
                and time_diff_seconds(
                    df.loc[i, "start_time"], df.loc[i + 1, "start_time"], today
                )
                < MIN_SUB_DUR
            ):
                df.loc[i, "text"] += " " + df.loc[i + 1, "text"]
                df.loc[i, "origin"] += " " + df.loc[i + 1, "origin"]
                df.loc[i, "end_time"] = df.loc[i + 1, "end_time"]
                df.loc[i, "raw_end_time"] = df.loc[i + 1, "raw_end_time"]
                df.loc[i, "duration"] = time_diff_seconds(
                    df.loc[i, "start_time"], df.loc[i, "end_time"], today
                )
                df = df.drop(i + 1).reset_index(drop=True)
            else:
                if i < len(df) - 1:  # Not the last audio
                    extended_dt = datetime.datetime.combine(
                        today, df.loc[i, "start_time"]
                    ) + datetime.timedelta(seconds=MIN_SUB_DUR)
                    next_start_dt = datetime.datetime.combine(
                        today, df.loc[i + 1, "start_time"]
                    )
                    clamped_dt = min(extended_dt, next_start_dt)
                    clamped_end = max(df.loc[i, "end_time"], clamped_dt.time())
                    df.loc[i, "end_time"] = clamped_end
                    df.loc[i, "duration"] = time_diff_seconds(
                        df.loc[i, "start_time"], df.loc[i, "end_time"], today
                    )
                i += 1
        else:
            i += 1

    df["start_time"] = df["start_time"].apply(lambda x: x.strftime("%H:%M:%S.%f")[:-3])
    df["end_time"] = df["end_time"].apply(lambda x: x.strftime("%H:%M:%S.%f")[:-3])
    df["raw_start_time"] = df["raw_start_time"].apply(
        lambda x: x.strftime("%H:%M:%S.%f")[:-3]
    )
    df["raw_end_time"] = df["raw_end_time"].apply(
        lambda x: x.strftime("%H:%M:%S.%f")[:-3]
    )

    return df


@check_file_exists("dub_tasks")
def gen_audio_task_main():
    df = process_srt()
    ws = get_active_workspace()
    ws.save_dub_tasks(df)
    report_progress(
        0.50,
        f"已构建 {len(df)} 条配音任务",
        f"配音时间轴任务构建完成（共 {len(df)} 条）",
        tag="asr",
        label="TTS",
    )


if __name__ == "__main__":
    gen_audio_task_main()

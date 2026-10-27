import datetime
import re
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.utils import check_file_exists, load_key, rprint
from core.utils.models import _8_1_AUDIO_TASK, resolve_task_path

console = Console()

TRANS_SUBS_FOR_AUDIO_FILE = "output/audio/trans_subs_for_audio.srt"
SRC_SUBS_FOR_AUDIO_FILE = "output/audio/src_subs_for_audio.srt"


def time_diff_seconds(t1, t2, base_date):
    """Calculate the difference in seconds between two time objects"""
    dt1 = datetime.datetime.combine(base_date, t1)
    dt2 = datetime.datetime.combine(base_date, t2)
    return (dt2 - dt1).total_seconds()


def process_srt():
    """Process srt file, generate audio tasks"""
    trans_subs_path = resolve_task_path(TRANS_SUBS_FOR_AUDIO_FILE)
    src_subs_path = resolve_task_path(SRC_SUBS_FOR_AUDIO_FILE)
    with open(trans_subs_path, "r", encoding="utf-8") as file:
        content = file.read()

    with open(src_subs_path, "r", encoding="utf-8") as src_file:
        src_content = src_file.read()

    subtitles = []
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
            text = " ".join(lines[2:])
            # Remove content within parentheses (including English and Chinese parentheses)
            text = re.sub(r"\([^)]*\)", "", text).strip()
            text = re.sub(r"（[^）]*）", "", text).strip()
            text = re.sub(r"\([^)]*$|（[^）]*$", "", text).strip()
            text = re.sub(r"^[^(]*\)|^[^（]*）", "", text).strip()
            # Remove dialogue dashes while preserving intra-word hyphens like "hands-on"
            text = re.sub(r"(?<!\w)-|-(?!\w)", "", text).strip()

            # Add the original text from src_subs_for_audio.srt
            origin = src_subtitles.get(number, "")

        except ValueError as e:
            rprint(
                Panel(
                    f"Unable to parse subtitle block '{block}', error: {str(e)}, skipping this subtitle block.",
                    title="Error",
                    border_style="red",
                )
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
                rprint(
                    f"[bold yellow]Merging subtitles {i + 1} and {i + 2}[/bold yellow]"
                )
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
                    rprint(
                        f"[bold blue]Extending subtitle {i + 1} duration to {MIN_SUB_DUR} seconds[/bold blue]"
                    )
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
                else:
                    rprint(
                        f"[bold red]The last subtitle {i + 1} duration is less than {MIN_SUB_DUR} seconds, but not extending[/bold red]"
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


@check_file_exists(_8_1_AUDIO_TASK)
def gen_audio_task_main():
    df = process_srt()
    console.print(df)
    out_path = resolve_task_path(_8_1_AUDIO_TASK)
    df.to_excel(out_path, index=False)
    rprint(
        Panel(
            f"Successfully generated {out_path}", title="Success", border_style="green"
        )
    )


if __name__ == "__main__":
    gen_audio_task_main()

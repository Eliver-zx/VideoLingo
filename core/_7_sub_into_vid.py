import os
import subprocess
import time

import cv2

from core._1_ytdlp import find_media_file, find_video_files as _default_find_video_files
from core.utils import load_key, rprint
from core.utils.models import _6_BI_ASS, resolve_task_path
from core.utils.subtitle_style import (
    build_ass_from_srt_pair,
    escape_ffmpeg_filter_path,
    get_burn_mode,
    get_optimal_video_encoder_args,
    get_subtitle_filters as _default_get_subtitle_filters,
)
from core.workspace import probe_media_metadata

find_video_files = _default_find_video_files
get_subtitle_filters = _default_get_subtitle_filters

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"


def check_gpu_available():
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def merge_subtitles_to_video():
    from core._1_ytdlp import is_audio_only_input

    if is_audio_only_input():
        rprint(
            "[bold green]🎵 Audio-only input: skipping video merge. "
            "Subtitle files are ready in the `output` directory.[/bold green]"
        )
        return

    if not load_key("burn_subtitles"):
        rprint(
            "[bold yellow]Subtitles burn-in is disabled (burn_subtitles=False); "
            "skipping subtitle video generation.[/bold yellow]"
        )
        return

    if find_video_files is not _default_find_video_files:
        video_file = find_video_files()
    else:
        media_file, media_type = find_media_file()
        if media_type != "video":
            rprint(
                "[bold green]🎵 Non-video input: skipping video merge. "
                "Subtitle files are ready in the `output` directory.[/bold green]"
            )
            return
        video_file = media_file

    out_video = resolve_task_path(OUTPUT_VIDEO)
    src_srt = resolve_task_path(SRC_SRT)
    trans_srt = resolve_task_path(TRANS_SRT)
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)

    if not os.path.exists(src_srt) or not os.path.exists(trans_srt):
        rprint("Subtitle files not found in the 'output' directory.")
        raise FileNotFoundError("Source or translation SRT file not found.")

    meta = probe_media_metadata(video_file)
    TARGET_WIDTH = int(meta.get("display_width") or meta.get("width") or 0)
    TARGET_HEIGHT = int(meta.get("display_height") or meta.get("height") or 0)
    if TARGET_WIDTH <= 0 or TARGET_HEIGHT <= 0:
        video = cv2.VideoCapture(video_file)
        TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
    if TARGET_WIDTH <= 0 or TARGET_HEIGHT <= 0:
        TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080

    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    if get_subtitle_filters is not _default_get_subtitle_filters:
        vf_filter = get_subtitle_filters("subtitle", src_srt, trans_srt)
    else:
        ass_path = resolve_task_path(_6_BI_ASS)
        build_ass_from_srt_pair(
            src_srt,
            trans_srt,
            ass_path,
            mode=get_burn_mode("subtitle"),
            display_width=TARGET_WIDTH,
            display_height=TARGET_HEIGHT,
        )
        vf_filter = f"ass='{escape_ffmpeg_filter_path(ass_path)}'"
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        video_file,
        "-vf",
        vf_filter,
    ]

    ffmpeg_gpu = bool(load_key("ffmpeg_gpu"))
    if ffmpeg_gpu:
        rprint("[bold green]will use GPU acceleration.[/bold green]")
    ffmpeg_cmd.extend(get_optimal_video_encoder_args(ffmpeg_gpu))
    ffmpeg_cmd.extend(["-c:a", "copy", "-y", out_video])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
        if process.returncode == 0:
            rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
        else:
            raise subprocess.CalledProcessError(process.returncode, ffmpeg_cmd)
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()
        raise


if __name__ == "__main__":
    merge_subtitles_to_video()

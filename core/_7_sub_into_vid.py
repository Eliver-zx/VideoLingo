import os
import subprocess
import time

import cv2

from core._1_ytdlp import find_media_file, find_video_files as _default_find_video_files
from core.utils import load_key, log_event, report_progress
from core.utils.models import resolve_task_path
from core.utils.subtitle_style import (
    get_optimal_video_encoder_args,
    get_subtitle_filters as _default_get_subtitle_filters,
    wrap_gpu_subtitle_filter,
)
from core.workspace import get_active_workspace, probe_media_metadata

find_video_files = _default_find_video_files
get_subtitle_filters = _default_get_subtitle_filters

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"


def _format_time_sec(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def check_gpu_available():
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return "h264_nvenc" in (result.stdout or "")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg executable not found in PATH. Please install FFmpeg "
            "(e.g. `winget install Gyan.FFmpeg` or `conda install ffmpeg`) and restart."
        ) from exc
    except Exception:
        return False


def merge_subtitles_to_video():
    from core._1_ytdlp import is_audio_only_input

    if is_audio_only_input():
        report_progress(
            1.0,
            "纯音频输入，跳过视频压制",
            "纯音频输入跳过视频压制，字幕已就绪",
            tag="asr",
            label="VIDEO",
        )
        return

    if not load_key("burn_subtitles"):
        report_progress(
            1.0,
            "未启用字幕烧录，跳过合成",
            "字幕硬烧录已关闭 (burn_subtitles=False)，跳过合成",
            tag="asr",
            label="VIDEO",
        )
        return

    if find_video_files is not _default_find_video_files:
        video_file = find_video_files()
    else:
        media_file, media_type = find_media_file()
        if media_type != "video":
            report_progress(
                1.0,
                "非视频输入，跳过视频压制",
                "非视频输入跳过视频压制，字幕已就绪",
                tag="asr",
                label="VIDEO",
            )
            return
        video_file = media_file

    out_video = resolve_task_path(OUTPUT_VIDEO)
    src_srt = resolve_task_path(SRC_SRT)
    trans_srt = resolve_task_path(TRANS_SRT)
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)

    if not os.path.exists(src_srt) or not os.path.exists(trans_srt):
        log_event(
            "未找到待压制的原文字幕或译文字幕文件",
            tag="warn",
            label="VIDEO",
            level="error",
        )
        raise FileNotFoundError("Source or translation SRT file not found.")

    from core._6_gen_sub import sync_subtitles_from_srt_if_newer

    sync_subtitles_from_srt_if_newer()

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

    vf_filter = get_subtitle_filters("subtitle", src_srt, trans_srt)
    ffmpeg_gpu = bool(load_key("ffmpeg_gpu"))
    hw_input_args, vf_filter = wrap_gpu_subtitle_filter(
        vf_filter, ffmpeg_gpu, video_meta=meta
    )
    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",
        "-progress",
        "pipe:1",
        "-nostats",
        *hw_input_args,
        "-i",
        video_file,
        "-vf",
        vf_filter,
    ]
    ffmpeg_cmd.extend(get_optimal_video_encoder_args(ffmpeg_gpu))
    ffmpeg_cmd.extend(["-c:a", "copy", "-movflags", "+faststart", "-y", out_video])

    gpu_label = "GPU" if ffmpeg_gpu else "CPU"
    total_duration = float(meta.get("duration_sec") or 0.0)
    report_progress(
        0.05,
        f"开始压制字幕 ({TARGET_WIDTH}x{TARGET_HEIGHT})",
        f"开始压制带字幕视频 ({TARGET_WIDTH}x{TARGET_HEIGHT}, {gpu_label})",
        tag="asr",
        label="VIDEO",
    )
    start_time = time.time()
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    stderr_chunks: list[str] = []

    def _drain_stderr():
        if process.stderr:
            for line in process.stderr:
                stderr_chunks.append(line)
                if len(stderr_chunks) > 100:
                    stderr_chunks.pop(0)

    import threading

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    last_report = 0.0
    cur_speed = ""
    cur_time_sec = 0.0

    try:
        if process.stdout:
            for raw_line in process.stdout:
                check_cancel()
                line = raw_line.strip()
                if not line:
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "out_time_us":
                        try:
                            cur_time_sec = max(0.0, float(v) / 1_000_000.0)
                        except ValueError:
                            pass
                    elif k == "speed":
                        cur_speed = v
                    elif k == "progress" and (v in ("continue", "end")):
                        now = time.time()
                        if (now - last_report >= 0.75) or (v == "end"):
                            last_report = now
                            if total_duration > 0:
                                frac = min(
                                    0.98, max(0.05, cur_time_sec / total_duration)
                                )
                                pct_int = int(round(frac * 100))
                                cur_str = _format_time_sec(cur_time_sec)
                                dur_str = _format_time_sec(total_duration)
                                speed_num = 0.0
                                if cur_speed.endswith("x"):
                                    try:
                                        speed_num = float(cur_speed[:-1].strip())
                                    except ValueError:
                                        pass
                                eta_str = ""
                                if speed_num > 0.05 and cur_time_sec < total_duration:
                                    remaining_sec = (
                                        total_duration - cur_time_sec
                                    ) / speed_num
                                    eta_str = (
                                        f" · 剩余 {_format_time_sec(remaining_sec)}"
                                    )
                                speed_str = (
                                    f" · 速度 {cur_speed}"
                                    if cur_speed and cur_speed != "N/A"
                                    else ""
                                )
                                sub_lbl = f"压制中 {pct_int}% (已处理 {cur_str}/{dur_str}{speed_str}{eta_str})"
                            else:
                                frac = 0.50
                                cur_str = _format_time_sec(cur_time_sec)
                                speed_str = (
                                    f" · 速度 {cur_speed}"
                                    if cur_speed and cur_speed != "N/A"
                                    else ""
                                )
                                sub_lbl = f"压制中 (已处理 {cur_str}{speed_str})"

                            report_progress(
                                frac,
                                sub_lbl,
                                tag="asr",
                                label="VIDEO",
                            )

        ret = process.wait()
        stderr_thread.join(timeout=2.0)
        if ret == 0:
            elapsed = time.time() - start_time
            get_active_workspace().notify_artifact_update(
                "video", tag="asr", label="VIDEO"
            )
            report_progress(
                1.0,
                f"视频压制完成 ({elapsed:.1f}s)",
                f"字幕视频压制完成，耗时 {elapsed:.1f}s",
                tag="asr",
                label="VIDEO",
            )
        else:
            err_tail = "".join(stderr_chunks).strip()[-800:]
            raise subprocess.CalledProcessError(ret, ffmpeg_cmd, stderr=err_tail)
    except Exception as e:
        detail = getattr(e, "stderr", "") or "".join(stderr_chunks).strip()[-800:]
        msg = f"字幕视频压制失败: {e}" + (f" | {detail}" if detail else "")
        log_event(
            msg,
            tag="warn",
            label="VIDEO",
            level="error",
        )
        if process.poll() is None:
            process.kill()
        raise


if __name__ == "__main__":
    merge_subtitles_to_video()

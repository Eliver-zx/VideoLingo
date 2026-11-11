from collections import deque
import contextlib
import os
import subprocess
import threading
import time
from typing import Callable

from core._1_ytdlp import find_media_file, find_video_files as _default_find_video_files
from core.task_runner import StopTask
from core.utils import check_cancel, load_key, log_event, report_progress
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

_MP4_COPY_AUDIO_CODECS = {"", "aac", "mp3", "opus", "ac3", "eac3", "alac", "mp2"}


def _format_time_sec(seconds: float) -> str:
    secs = max(0, int(round(seconds)))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _resolve_video_metadata(video_file: str) -> dict:
    ws = get_active_workspace()
    if ws.manifest_path.is_file():
        manifest = ws.load_manifest()
        media = manifest.get("media") or {}
        fname = media.get("filename")
        if (
            fname
            and int(media.get("width") or 0) > 0
            and int(media.get("height") or 0) > 0
            and float(media.get("duration_sec") or 0.0) > 0
            and "audio_codec" in media
        ):
            cand = (ws.root / fname).resolve()
            if os.path.abspath(video_file) == str(cand):
                return media
    return probe_media_metadata(video_file)


def burn_subtitles_to_video(
    video_file: str,
    out_video: str,
    vf_filter: str,
    *,
    ffmpeg_gpu: bool | None = None,
    on_progress: Callable[[float, str, str | None], None] | None = None,
    on_log: Callable[[str, str, str, bool], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    register_process: Callable[[subprocess.Popen[str] | None], None] | None = None,
) -> float:
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    use_gpu = bool(load_key("ffmpeg_gpu")) if ffmpeg_gpu is None else bool(ffmpeg_gpu)

    def _emit_progress(frac: float, sub_lbl: str, msg: str | None = None) -> None:
        if on_progress is not None:
            on_progress(frac, sub_lbl, msg)
        else:
            report_progress(frac, sub_lbl, msg, tag="asr", label="VIDEO")

    def _emit_error_log(msg: str) -> None:
        if on_log is not None:
            on_log(msg, "warn", "VIDEO", True)
        else:
            log_event(msg, tag="warn", label="VIDEO", level="error")

    def _check_abort() -> None:
        if should_stop is not None:
            if should_stop():
                raise StopTask()
        else:
            check_cancel()

    _check_abort()
    meta = _resolve_video_metadata(video_file)
    target_width = int(meta.get("display_width") or meta.get("width") or 0)
    target_height = int(meta.get("display_height") or meta.get("height") or 0)
    if target_width <= 0 or target_height <= 0:
        import cv2

        video = cv2.VideoCapture(video_file)
        target_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
    if target_width <= 0 or target_height <= 0:
        target_width, target_height = 1920, 1080

    hw_input_args, wrapped_vf = wrap_gpu_subtitle_filter(
        vf_filter, use_gpu, video_meta=meta
    )
    audio_codec = str(meta.get("audio_codec") or "").lower()
    audio_args = (
        ["-c:a", "copy"]
        if audio_codec in _MP4_COPY_AUDIO_CODECS
        else ["-c:a", "aac", "-b:a", "192k"]
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
        wrapped_vf,
    ]
    ffmpeg_cmd.extend(get_optimal_video_encoder_args(use_gpu))
    ffmpeg_cmd.extend([*audio_args, "-movflags", "+faststart", "-y", out_video])

    _check_abort()
    gpu_label = "GPU" if use_gpu else "CPU"
    total_duration = float(meta.get("duration_sec") or 0.0)
    _emit_progress(
        0.05,
        f"开始压制字幕 ({target_width}x{target_height})",
        f"开始压制带字幕视频 ({target_width}x{target_height}, {gpu_label})",
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
    stderr_thread: threading.Thread | None = None
    stderr_chunks: deque[str] = deque(maxlen=100)

    def _terminate_and_reap() -> None:
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=5.0)
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=2.0)

    try:
        if register_process is not None:
            register_process(process)
        _check_abort()

        def _drain_stderr():
            if process.stderr:
                for line in process.stderr:
                    stderr_chunks.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        last_report = 0.0
        cur_speed = ""
        cur_time_sec = 0.0

        if process.stdout:
            for raw_line in process.stdout:
                _check_abort()
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
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
                            frac = min(0.98, max(0.05, cur_time_sec / total_duration))
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
                                eta_str = f" · 剩余 {_format_time_sec(remaining_sec)}"
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

                        _emit_progress(frac, sub_lbl, None)

        ret = process.wait()
        stderr_thread.join(timeout=2.0)
        _check_abort()
        if ret != 0:
            err_tail = "".join(stderr_chunks).strip()[-800:]
            raise subprocess.CalledProcessError(ret, ffmpeg_cmd, stderr=err_tail)
        return time.time() - start_time
    except StopTask:
        _terminate_and_reap()
        raise
    except Exception as e:
        _terminate_and_reap()
        detail = getattr(e, "stderr", "") or "".join(stderr_chunks).strip()[-800:]
        msg = f"字幕视频压制失败: {e}" + (f" | {detail}" if detail else "")
        _emit_error_log(msg)
        raise
    finally:
        if register_process is not None:
            register_process(None)
        if process.stdout is not None:
            with contextlib.suppress(OSError):
                process.stdout.close()
        if process.stderr is not None:
            with contextlib.suppress(OSError):
                process.stderr.close()


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

    vf_filter = get_subtitle_filters("subtitle", src_srt, trans_srt)
    elapsed = burn_subtitles_to_video(video_file, out_video, vf_filter)
    get_active_workspace().notify_artifact_update("video", tag="asr", label="VIDEO")
    report_progress(
        1.0,
        f"视频压制完成 ({elapsed:.1f}s)",
        f"字幕视频压制完成，耗时 {elapsed:.1f}s",
        tag="asr",
        label="VIDEO",
    )


if __name__ == "__main__":
    merge_subtitles_to_video()

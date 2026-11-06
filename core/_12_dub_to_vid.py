import os
from pathlib import Path
import subprocess
import json
import math

import cv2

from core._1_ytdlp import find_media_file, find_video_files as _ytdlp_find_video_files
from core.utils import check_cancel, load_key, log_event, report_progress
from core.utils.models import _BACKGROUND_AUDIO_FILE, _DUB_WAV_FILE, resolve_task_path
from core.utils.subtitle_style import (
    build_ass_from_srt_pair,
    escape_ffmpeg_filter_path,
    get_burn_mode,
    get_optimal_video_encoder_args,
    get_subtitle_filters as _default_get_subtitle_filters,
    wrap_gpu_subtitle_filter,
)
from core.workspace import get_active_workspace, probe_media_metadata

find_video_files = _ytdlp_find_video_files
get_subtitle_filters = _default_get_subtitle_filters

DUB_VIDEO = "output/output_dub.mp4"
SUB_VIDEO = "output/output_sub.mp4"
DUB_SUB_FILE = "output/dub.srt"
DUB_SRC_SUB_FILE = "output/dub_src.srt"
DUB_AUDIO = "output/dub.mp3"


def normalize_dub_audio(audio_path, output_path):
    """Measure gated loudness, then apply one peak-limited gain to the whole dub."""
    check_cancel()
    measured = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(audio_path),
            "-af",
            "loudnorm=I=-20:TP=-1:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # FFmpeg can write its final progress summary after the measurement object.
    stats, _ = json.JSONDecoder().raw_decode(
        measured.stderr[measured.stderr.rfind("{") :]
    )
    loudness, peak = float(stats["input_i"]), float(stats["input_tp"])
    # Silence/very short clips may not have a measurable integrated loudness.
    # Preserve their level rather than applying an infinite or guessed gain.
    gain = (
        min(-20.0 - loudness, -1.0 - peak)
        if math.isfinite(loudness) and math.isfinite(peak)
        else 0.0
    )
    check_cancel()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(audio_path),
            "-af",
            f"volume={gain:.8f}dB",
            "-c:a",
            "pcm_s24le",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    from core._1_ytdlp import is_audio_only_input

    if is_audio_only_input():
        report_progress(
            1.0,
            "纯音频输入，跳过配音视频合成",
            "纯音频输入跳过配音视频封装，配音音频已就绪",
            tag="asr",
            label="VIDEO",
        )
        return

    VIDEO_FILE = (
        find_video_files()
        if find_video_files is not _ytdlp_find_video_files
        else find_media_file()[0]
    )
    background_file = resolve_task_path(_BACKGROUND_AUDIO_FILE)
    dub_video = resolve_task_path(DUB_VIDEO)
    sub_video = resolve_task_path(SUB_VIDEO)
    dub_sub_file = resolve_task_path(DUB_SUB_FILE)
    dub_src_sub_file = resolve_task_path(DUB_SRC_SUB_FILE)
    dub_audio_mp3 = resolve_task_path(DUB_AUDIO)
    dub_audio_wav = resolve_task_path(_DUB_WAV_FILE)
    trans_srt_file = resolve_task_path("output/trans.srt")

    burn_subtitles = load_key("burn_subtitles")

    # Prefer lossless dub.wav when available, falling back to dub.mp3
    dub_source = dub_audio_wav if os.path.isfile(dub_audio_wav) else dub_audio_mp3
    normalized_dub_audio = resolve_task_path("output/normalized_dub.wav")
    os.makedirs(os.path.dirname(normalized_dub_audio) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(dub_video) or ".", exist_ok=True)
    report_progress(
        0.15,
        "校准配音音频响度…",
        "正在执行配音音轨响度均衡 (EBU R128 loudnorm)",
        tag="asr",
        label="AUDIO",
    )
    normalize_dub_audio(dub_source, normalized_dub_audio)

    has_background = os.path.isfile(background_file)
    same_sub_timeline = (
        os.path.isfile(dub_sub_file)
        and os.path.isfile(trans_srt_file)
        and Path(dub_sub_file).read_text(encoding="utf-8", errors="replace")
        == Path(trans_srt_file).read_text(encoding="utf-8", errors="replace")
    )
    can_reuse_sub_video = (
        burn_subtitles
        and get_subtitle_filters is _default_get_subtitle_filters
        and os.path.isfile(sub_video)
        and os.path.getsize(sub_video) > 0
        and get_burn_mode("dubbed") == get_burn_mode("subtitle")
        and same_sub_timeline
    )

    input_video = sub_video if can_reuse_sub_video else VIDEO_FILE
    use_gpu = bool(load_key("ffmpeg_gpu"))

    filters = []
    hw_input_args: list[str] = []
    need_burn_filter = burn_subtitles and not can_reuse_sub_video
    if need_burn_filter:
        meta = probe_media_metadata(VIDEO_FILE)
        TARGET_WIDTH = int(meta.get("display_width") or meta.get("width") or 0)
        TARGET_HEIGHT = int(meta.get("display_height") or meta.get("height") or 0)
        if TARGET_WIDTH <= 0 or TARGET_HEIGHT <= 0:
            video = cv2.VideoCapture(VIDEO_FILE)
            TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video.release()
        if TARGET_WIDTH <= 0 or TARGET_HEIGHT <= 0:
            TARGET_WIDTH, TARGET_HEIGHT = 1920, 1080

        if (
            get_subtitle_filters is not _default_get_subtitle_filters
            or not os.path.isfile(dub_sub_file)
        ):
            subtitle_filter = get_subtitle_filters(
                "dubbed", dub_src_sub_file, dub_sub_file
            )
        else:
            dub_ass_path = resolve_task_path("output/dub_bi.ass")
            build_ass_from_srt_pair(
                dub_src_sub_file,
                dub_sub_file,
                dub_ass_path,
                mode=get_burn_mode("dubbed"),
                display_width=TARGET_WIDTH,
                display_height=TARGET_HEIGHT,
            )
            subtitle_filter = f"ass='{escape_ffmpeg_filter_path(dub_ass_path)}'"
        hw_input_args, subtitle_filter = wrap_gpu_subtitle_filter(
            subtitle_filter, use_gpu, video_meta=meta
        )
        filters.append(f"[0:v]{subtitle_filter}[v]")
    elif can_reuse_sub_video:
        log_event(
            "复用已压制字幕视频流，直接封装配音音轨",
            tag="asr",
            label="VIDEO",
        )
    cmd = ["ffmpeg", "-nostdin", "-y", *hw_input_args, "-i", input_video]
    if has_background:
        cmd.extend(["-i", background_file])
    cmd.extend(["-i", normalized_dub_audio])
    if has_background:
        filters.append(
            "[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]"
        )
    else:
        log_event(
            "未检测到独立背景音轨，仅导出配音人声音轨",
            tag="warn",
            label="AUDIO",
            level="warn",
        )
    if filters:
        cmd.extend(["-filter_complex", ";".join(filters)])

    cmd.extend(
        [
            "-map",
            "[v]" if need_burn_filter else "0:v:0",
            "-map",
            "[a]" if has_background else "1:a",
        ]
    )
    audio_output = ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dub_video]

    report_progress(
        0.45,
        "合成配音视频中…",
        "开始合成最终配音成片视频",
        tag="asr",
        label="VIDEO",
    )
    if need_burn_filter:
        encoder = get_optimal_video_encoder_args(use_gpu)
        subprocess.run(cmd + encoder + audio_output, check=True)
    else:
        try:
            # The picture is not touched, so it is copied as it is
            subprocess.run(cmd + ["-c:v", "copy"] + audio_output, check=True)
        except subprocess.CalledProcessError:
            log_event(
                "原视频流无法直接拷贝至 MP4，回退至重编码封装",
                tag="warn",
                label="VIDEO",
                level="warn",
            )
            encoder = get_optimal_video_encoder_args(use_gpu)
            subprocess.run(cmd + encoder + audio_output, check=True)
    get_active_workspace().notify_artifact_update("video", tag="asr", label="VIDEO")
    report_progress(
        1.0,
        "配音成片合成完成",
        f"配音成片视频已生成 ({dub_video})",
        tag="asr",
        label="VIDEO",
    )


if __name__ == "__main__":
    merge_video_audio()

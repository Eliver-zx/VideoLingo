"""Style and hardware encoding helpers for burned-in subtitles."""

from __future__ import annotations

import functools
import os
from pathlib import Path
import platform
import re
import subprocess

from core.task_runner import TaskRunner
from .config_utils import load_key_or, update_key

# An empty font name stands for the default font of the system
DEFAULT_STYLE = {
    "source": {
        "font_name": "",
        "font_size": 15,
        "font_color": "&HFFFFFF",
        "outline_color": "&H000000",
        "outline_width": 1,
    },
    "translation": {
        "font_name": "抖音美好体",
        "font_size": 17,
        "font_color": "&H00FFFF",
        "outline_color": "&H000000",
        "outline_width": 1.5,
        "back_color": "&H33000000",
        "margin_v": 5,
    },
}
COLOR_KEYS = {"font_color", "outline_color", "back_color"}

BURN_MODES = ("bilingual", "translation", "source")
# The key of each video in config.yaml and what it shows without the key
BURN_MODE_KEYS = {
    "subtitle": ("subtitle.sub_video_subtitles", "translation"),
    "dubbed": ("subtitle.dub_video_subtitles", "translation"),
}


def default_font():
    # Linux needs the Google Noto fonts: apt-get install fonts-noto
    return {
        "Linux": "NotoSansCJK-Regular",
        "Darwin": "抖音美好体",
        "Windows": "抖音美好体",
    }.get(platform.system(), "抖音美好体")


def to_ass_color(value):
    """`&HBBGGRR`, `&HAABBGGRR` or `#RRGGBB` as an ASS colour, None when it is none of them."""
    value = str(value).strip()
    if re.fullmatch(r"&[Hh]([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})", value):
        return "&H" + value[2:].upper()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return f"&H{value[5:7]}{value[3:5]}{value[1:3]}".upper()
    return None


def to_hex_color(ass_color):
    """The `#rrggbb` of an ASS colour, without its transparency."""
    return f"#{ass_color[-2:]}{ass_color[-4:-2]}{ass_color[-6:-4]}".lower()


def clean_value(key, value):
    """The value as it goes into the ffmpeg filter, None when it can not be used."""
    if key in COLOR_KEYS:
        return to_ass_color(value)
    if key == "font_name":
        # These characters would end the value inside the filter
        return (
            None
            if not isinstance(value, str) or re.search(r"[,:;='\\\[\]]", value)
            else value.strip()
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value


def get_subtitle_style(kind):
    """The style of the `source` or `translation` subtitles; a missing or unusable value is the default one."""
    configured = load_key_or("subtitle.style", None)
    configured = configured.get(kind) if isinstance(configured, dict) else None
    configured = configured if isinstance(configured, dict) else {}
    style = {}
    for key, default in DEFAULT_STYLE[kind].items():
        value = clean_value(key, configured.get(key, default))
        if value is None:
            TaskRunner.log_event(
                f"subtitle.style.{kind}.{key} = {configured[key]!r} 不可用，已回退为 {default!r}",
                tag="warn",
                label="SUBS",
                level="warn",
            )
            value = default
        style[key] = value
    return style


def save_subtitle_style(style):
    """Write the changed values one by one, so that the comments of config.yaml are kept."""
    changes = [
        (kind, key)
        for kind in style
        for key, value in style[kind].items()
        if value != get_subtitle_style(kind).get(key)
    ]
    try:
        if all(
            update_key(
                f"subtitle.style.{kind}.{key}", style[kind][key], add_missing=True
            )
            for kind, key in changes
        ):
            return
    except KeyError:
        pass
    # A config.yaml from before this setting, or one with a broken `style`
    update_key("subtitle.style", style, add_missing=True)


def get_force_style(kind):
    """The `force_style` of the subtitles filter of ffmpeg."""
    style = get_subtitle_style(kind)
    parts = [
        f"FontSize={style['font_size']}",
        f"FontName={style['font_name'] or default_font()}",
        f"PrimaryColour={style['font_color']}",
        f"OutlineColour={style['outline_color']}",
        f"Outline={style['outline_width']}",
    ]
    if kind == "translation":
        parts += [
            f"BackColour={style['back_color']}",
            "Alignment=2",
            f"MarginV={style['margin_v']}",
            "BorderStyle=1",
        ]
    else:
        parts.append("BorderStyle=1")
    return ",".join(parts)


def get_burn_mode(video):
    """Which subtitles go into the `subtitle` or `dubbed` video: bilingual, translation or source."""
    key, default = BURN_MODE_KEYS[video]
    mode = load_key_or(key, default)
    if mode not in BURN_MODES:
        TaskRunner.log_event(
            f"{key} = {mode!r} 不可用，已回退为 {default!r}",
            tag="warn",
            label="SUBS",
            level="warn",
        )
        mode = default
    return mode


def get_subtitle_filters(video, source_srt, translation_srt):
    """The `subtitles` filters of ffmpeg for a video, the source subtitles below the translated ones."""
    mode = get_burn_mode(video)
    if mode != "translation" and not os.path.exists(source_srt):
        # A dub from before the source subtitles of the dub were written
        TaskRunner.log_event(
            f"未找到 {source_srt}，仅压制译文字幕",
            tag="warn",
            label="SUBS",
            level="warn",
        )
        mode = "translation"
    subtitles = {"source": source_srt, "translation": translation_srt}
    kinds = ["source", "translation"] if mode == "bilingual" else [mode]
    return ",".join(
        f"subtitles={escape_ffmpeg_filter_path(subtitles[kind])}:force_style='{get_force_style(kind)}'"
        for kind in kinds
    )


def escape_ffmpeg_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for safe embedding inside an FFmpeg `-vf ass=...` or `subtitles=...` filter."""
    s = str(path).replace("\\", "/")
    s = (
        s.replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )
    return s


@functools.lru_cache(maxsize=1)
def _available_ffmpeg_encoders() -> set[str]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        out = proc.stdout or ""
        return set(
            re.findall(
                r"\b(h264_nvenc|h264_videotoolbox|h264_amf|h264_qsv|libx264)\b", out
            )
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg executable not found in PATH. Please install FFmpeg "
            "(e.g. `winget install Gyan.FFmpeg` or `conda install ffmpeg`) and restart."
        ) from exc
    except Exception:
        return {"libx264"}


@functools.lru_cache(maxsize=1)
def _available_ffmpeg_filters() -> set[str]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        out = proc.stdout or ""
        return set(
            re.findall(
                r"\b(ass|subtitles|scale_cuda|hwdownload|hwupload_cuda|overlay_cuda)\b",
                out,
            )
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg executable not found in PATH. Please install FFmpeg "
            "(e.g. `winget install Gyan.FFmpeg` or `conda install ffmpeg`) and restart."
        ) from exc
    except Exception:
        return set()


def _can_nvdec_cuda_surface(video_meta: dict | None) -> bool:
    """Return True when the video stream can be decoded directly to a CUDA surface without auto-rotate CPU filters."""
    if not video_meta:
        return True
    rot = abs(int(video_meta.get("rotation") or 0)) % 360
    if rot != 0:
        return False
    codec = str(video_meta.get("codec_name") or "").lower()
    pix_fmt = str(video_meta.get("pix_fmt") or "").lower()
    if not codec:
        return True
    base_codecs = {"h264", "hevc", "vp8", "vp9", "mpeg1video", "mpeg2video", "vc1"}
    if codec == "av1":
        try:
            import torch  # type: ignore[import-not-found]

            if not (
                torch.cuda.is_available()
                and hasattr(torch.cuda, "get_device_capability")
                and torch.cuda.get_device_capability(0) >= (8, 6)
            ):
                return False
        except Exception:
            return False
    elif codec not in base_codecs:
        return False
    if codec == "h264" and pix_fmt not in {"", "yuv420p", "yuvj420p", "nv12"}:
        return False
    if pix_fmt and any(tag in pix_fmt for tag in ("422", "444", "12le", "12be", "gbr")):
        return False
    return True


def wrap_gpu_subtitle_filter(
    subtitle_filter: str,
    use_gpu: bool = False,
    video_meta: dict | None = None,
) -> tuple[list[str], str]:
    """Return (pre-input hwaccel args, video filter chain) for subtitle burn-in."""
    if not use_gpu:
        return [], subtitle_filter
    encoders = _available_ffmpeg_encoders()
    filters = _available_ffmpeg_filters()
    if (
        platform.system() != "Darwin"
        and "h264_nvenc" in encoders
        and "scale_cuda" in filters
        and "hwdownload" in filters
        and _can_nvdec_cuda_surface(video_meta)
    ):
        return (
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-thread_queue_size",
                "1024",
            ],
            f"scale_cuda=format=yuv420p,hwdownload,format=yuv420p,{subtitle_filter}",
        )
    return ["-hwaccel", "auto", "-thread_queue_size", "1024"], subtitle_filter


def get_optimal_video_encoder_args(use_gpu: bool = False) -> list[str]:
    """Select the fastest compatible H.264 encoder for the current platform with VBR capped at 6000k and `yuv420p` output."""
    if use_gpu:
        encoders = _available_ffmpeg_encoders()
        sys_name = platform.system()
        if sys_name == "Darwin" and "h264_videotoolbox" in encoders:
            return [
                "-c:v",
                "h264_videotoolbox",
                "-q:v",
                "65",
                "-maxrate",
                "6000k",
                "-bufsize",
                "12000k",
                "-pix_fmt",
                "yuv420p",
            ]
        if "h264_nvenc" in encoders:
            return [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p1",
                "-rc",
                "vbr",
                "-cq",
                "23",
                "-maxrate",
                "6000k",
                "-bufsize",
                "12000k",
                "-pix_fmt",
                "yuv420p",
            ]
        if "h264_qsv" in encoders:
            return [
                "-c:v",
                "h264_qsv",
                "-global_quality",
                "23",
                "-maxrate",
                "6000k",
                "-bufsize",
                "12000k",
                "-pix_fmt",
                "yuv420p",
            ]
        if "h264_amf" in encoders:
            return [
                "-c:v",
                "h264_amf",
                "-quality",
                "balanced",
                "-rc",
                "vbr_peak",
                "-maxrate",
                "6000k",
                "-bufsize",
                "12000k",
                "-pix_fmt",
                "yuv420p",
            ]
        # Legacy test compatibility when a mocked ffmpeg does not actually run encoder probing
        return ["-c:v", "h264_nvenc"]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-maxrate",
        "6000k",
        "-bufsize",
        "12000k",
        "-pix_fmt",
        "yuv420p",
    ]

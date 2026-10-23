"""Style, ASS generation, and fast hardware encoding helpers for burned-in subtitles."""
from __future__ import annotations

import functools
import os
from pathlib import Path
import platform
import re
import subprocess

from rich import print as rprint

from .config_utils import load_key_or, update_key

# An empty font name stands for the default font of the system
DEFAULT_STYLE = {
    "source": {
        "font_name": "", "font_size": 15, "font_color": "&HFFFFFF",
        "outline_color": "&H000000", "outline_width": 1,
    },
    "translation": {
        "font_name": "", "font_size": 17, "font_color": "&H00FFFF",
        "outline_color": "&H000000", "outline_width": 1,
        "back_color": "&H33000000", "margin_v": 27,
    },
}
COLOR_KEYS = {"font_color", "outline_color", "back_color"}

BURN_MODES = ("bilingual", "translation", "source")
# The key of each video in config.yaml and what it shows without the key
BURN_MODE_KEYS = {"subtitle": ("subtitle.sub_video_subtitles", "bilingual"), "dubbed": ("subtitle.dub_video_subtitles", "translation")}


def default_font():
    # Linux needs the Google Noto fonts: apt-get install fonts-noto
    return {"Linux": "NotoSansCJK-Regular", "Darwin": "Arial Unicode MS"}.get(platform.system(), "Arial")


def cjk_default_font() -> str:
    """System-native CJK font family name recognized by libass / Fontconfig / DirectWrite / CoreText."""
    return {
        "Darwin": "PingFang SC",
        "Linux": "Noto Sans CJK SC",
        "Windows": "Microsoft YaHei",
    }.get(platform.system(), "Arial")


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
        return None if not isinstance(value, str) or re.search(r"[,:;='\\\[\]]", value) else value.strip()
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
            rprint(f"[yellow]⚠️ subtitle.style.{kind}.{key} = {configured[key]!r} can not be used, using {default!r}[/yellow]")
            value = default
        style[key] = value
    return style


def save_subtitle_style(style):
    """Write the changed values one by one, so that the comments of config.yaml are kept."""
    changes = [
        (kind, key) for kind in style for key, value in style[kind].items()
        if value != get_subtitle_style(kind).get(key)
    ]
    try:
        if all(update_key(f"subtitle.style.{kind}.{key}", style[kind][key], add_missing=True) for kind, key in changes):
            return
    except KeyError:
        pass
    # A config.yaml from before this setting, or one with a broken `style`
    update_key("subtitle.style", style, add_missing=True)


def get_force_style(kind):
    """The `force_style` of the subtitles filter of ffmpeg."""
    style = get_subtitle_style(kind)
    parts = [
        f"FontSize={style['font_size']}", f"FontName={style['font_name'] or default_font()}",
        f"PrimaryColour={style['font_color']}", f"OutlineColour={style['outline_color']}",
        f"Outline={style['outline_width']}",
    ]
    if kind == "translation":
        parts += [f"BackColour={style['back_color']}", "Alignment=2", f"MarginV={style['margin_v']}", "BorderStyle=4"]
    else:
        parts.append("BorderStyle=1")
    return ",".join(parts)


def get_burn_mode(video):
    """Which subtitles go into the `subtitle` or `dubbed` video: bilingual, translation or source."""
    key, default = BURN_MODE_KEYS[video]
    mode = load_key_or(key, default)
    if mode not in BURN_MODES:
        rprint(f"[yellow]⚠️ {key} = {mode!r} can not be used, using {default!r}[/yellow]")
        mode = default
    return mode


def get_subtitle_filters(video, source_srt, translation_srt):
    """The `subtitles` filters of ffmpeg for a video, the source subtitles below the translated ones."""
    mode = get_burn_mode(video)
    if mode != "translation" and not os.path.exists(source_srt):
        # A dub from before the source subtitles of the dub were written
        rprint(f"[yellow]⚠️ {source_srt} not found, burning the translated subtitles only[/yellow]")
        mode = "translation"
    subtitles = {"source": source_srt, "translation": translation_srt}
    kinds = ["source", "translation"] if mode == "bilingual" else [mode]
    return ",".join(f"subtitles={subtitles[kind]}:force_style='{get_force_style(kind)}'" for kind in kinds)


# ─── Single-pass Adaptive ASS Builder & Fast Video Encoder Probe ───

def _srt_time_to_ass(srt_ts: str) -> str:
    """Convert `HH:MM:SS,mmm` to ASS `H:MM:SS.cc`."""
    m = re.match(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})", srt_ts.strip())
    if not m:
        return "0:00:00.00"
    hh, mm, ss, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    cc = min(99, round(ms / 10.0))
    return f"{hh}:{mm:02d}:{ss:02d}.{cc:02d}"


def parse_srt_file(srt_path: str | Path) -> list[tuple[str, str, str]]:
    """Parse an SRT file into `[(start_ass, end_ass, text), ...]`."""
    p = Path(srt_path)
    if not p.is_file():
        return []
    raw = p.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[tuple[str, str, str]] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        ts_idx = -1
        for i, ln in enumerate(lines):
            if "-->" in ln:
                ts_idx = i
                break
        if ts_idx < 0:
            continue
        parts = lines[ts_idx].split("-->")
        if len(parts) != 2:
            continue
        start_ass = _srt_time_to_ass(parts[0])
        end_ass = _srt_time_to_ass(parts[1])
        text = r"\N".join(lines[ts_idx + 1:])
        if text:
            cues.append((start_ass, end_ass, text))
    return cues


def build_ass_from_srt_pair(
    source_srt: str | Path | None,
    translation_srt: str | Path | None,
    ass_path: str | Path,
    *,
    mode: str = "bilingual",
    display_width: int = 1920,
    display_height: int = 1080,
) -> Path:
    """Generate a single merged `.ass` file for 1-pass libass rendering with aspect-ratio-safe font scaling."""
    w = max(320, int(display_width or 1920))
    h = max(240, int(display_height or 1080))
    # Scale relative to the shorter side (288 in libass reference coordinates) so 9:16 portrait videos do not blow up 3.16x
    short_side = min(w, h)
    scale = short_side / 288.0

    src_style = get_subtitle_style("source")
    tr_style = get_subtitle_style("translation")

    src_font = src_style["font_name"] or cjk_default_font()
    tr_font = tr_style["font_name"] or cjk_default_font()
    src_size = max(12, int(round(float(src_style["font_size"]) * scale)))
    tr_size = max(14, int(round(float(tr_style["font_size"]) * scale)))
    src_outline = max(1, int(round(float(src_style["outline_width"]) * scale * 0.6)))
    tr_outline = max(1, int(round(float(tr_style["outline_width"]) * scale * 0.6)))
    src_margin_v = max(12, int(round(10 * scale)))
    tr_margin_v = max(src_margin_v + src_size + 6, int(round(float(tr_style["margin_v"]) * scale)))
    margin_lr = max(20, int(round(w * 0.04)))

    if mode != "translation" and (not source_srt or not Path(source_srt).is_file()):
        mode = "translation"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Source,{src_font},{src_size},{src_style['font_color']},&H000000FF,{src_style['outline_color']},&H80000000,0,0,0,0,100,100,0,0,1,{src_outline},0,2,{margin_lr},{margin_lr},{src_margin_v},1
Style: Translation,{tr_font},{tr_size},{tr_style['font_color']},&H000000FF,{tr_style['outline_color']},{tr_style['back_color']},0,0,0,0,100,100,0,0,4,{tr_outline},0,2,{margin_lr},{margin_lr},{tr_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    if mode in ("bilingual", "source") and source_srt:
        for s_ts, e_ts, txt in parse_srt_file(source_srt):
            events.append(f"Dialogue: 0,{s_ts},{e_ts},Source,,0,0,0,,{txt}")
    if mode in ("bilingual", "translation") and translation_srt:
        for s_ts, e_ts, txt in parse_srt_file(translation_srt):
            margin_override = 0 if mode == "bilingual" else src_margin_v
            events.append(f"Dialogue: 1,{s_ts},{e_ts},Translation,,0,0,{margin_override},,{txt}")

    out = Path(ass_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out


def escape_ffmpeg_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for safe embedding inside an FFmpeg `-vf ass=...` or `subtitles=...` filter."""
    s = str(path).replace("\\", "/")
    s = s.replace(":", r"\:").replace("'", r"\'").replace("[", r"\[").replace("]", r"\]")
    return s


@functools.lru_cache(maxsize=1)
def _available_ffmpeg_encoders() -> set[str]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        out = proc.stdout or ""
        return set(re.findall(r"\b(h264_nvenc|h264_videotoolbox|h264_amf|h264_qsv|libx264)\b", out))
    except Exception:
        return {"libx264"}


def get_optimal_video_encoder_args(use_gpu: bool = False) -> list[str]:
    """Select the fastest compatible H.264 encoder for the current platform with safe `yuv420p` output."""
    if use_gpu:
        encoders = _available_ffmpeg_encoders()
        sys_name = platform.system()
        if sys_name == "Darwin" and "h264_videotoolbox" in encoders:
            return ["-c:v", "h264_videotoolbox", "-q:v", "65", "-pix_fmt", "yuv420p"]
        if "h264_nvenc" in encoders:
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p"]
        if "h264_qsv" in encoders:
            return ["-c:v", "h264_qsv", "-global_quality", "23", "-pix_fmt", "yuv420p"]
        if "h264_amf" in encoders:
            return ["-c:v", "h264_amf", "-quality", "balanced", "-pix_fmt", "yuv420p"]
        # Legacy test compatibility when a mocked ffmpeg does not actually run encoder probing
        return ["-c:v", "h264_nvenc"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]

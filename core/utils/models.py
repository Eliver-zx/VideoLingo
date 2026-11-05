from __future__ import annotations

from pathlib import Path

# ------------------------------------------
# Audio, Subtitle & Log Paths
# ------------------------------------------
_10_DUB_TRUNCATED = "output/log/dub_truncated.json"

_OUTPUT_DIR = "output"
_AUDIO_DIR = "output/audio"
_RAW_AUDIO_FILE = "output/audio/raw.mp3"
_VOCAL_AUDIO_FILE = "output/audio/vocal.mp3"
_BACKGROUND_AUDIO_FILE = "output/audio/background.mp3"
_DUB_WAV_FILE = "output/audio/dub.wav"
_BI_SRT = "output/bi.srt"
_BI_ASS = "output/bi.ass"
_6_BI_ASS = _BI_ASS
_AUDIO_REFERS_DIR = "output/audio/refers"
_AUDIO_SEGS_DIR = "output/audio/segs"
_AUDIO_TMP_DIR = "output/audio/tmp"

# ------------------------------------------
# Done markers
# ------------------------------------------
_TEXT_DONE_MARKER = "output/.subtitle_done"
_AUDIO_DONE_MARKER = "output/.dubbing_done"


def resolve_task_path(path_str: str | Path) -> str:
    """Resolve a default `output/...` path into the active `TaskWorkspace` while honoring test monkeypatches."""
    from core.workspace import get_active_workspace

    s = str(path_str).replace("\\", "/")
    ws = get_active_workspace()
    ws_root = str(ws.root).replace("\\", "/")
    if ws_root == "output":
        return s
    if s == "output":
        return ws_root
    if s.startswith("output/"):
        return f"{ws_root}/{s[len('output/') :]}"
    return s


__all__ = [
    "_10_DUB_TRUNCATED",
    "_OUTPUT_DIR",
    "_AUDIO_DIR",
    "_RAW_AUDIO_FILE",
    "_VOCAL_AUDIO_FILE",
    "_BACKGROUND_AUDIO_FILE",
    "_DUB_WAV_FILE",
    "_BI_SRT",
    "_BI_ASS",
    "_6_BI_ASS",
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR",
    "_TEXT_DONE_MARKER",
    "_AUDIO_DONE_MARKER",
    "resolve_task_path",
]

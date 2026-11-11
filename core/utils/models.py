from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
import subprocess
import sys

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
_AUDIO_REFERS_DIR = "output/audio/refers"
_AUDIO_SEGS_DIR = "output/audio/segs"
_AUDIO_TMP_DIR = "output/audio/tmp"

# ------------------------------------------
# Done markers
# ------------------------------------------
_TEXT_DONE_MARKER = "output/.subtitle_done"
_AUDIO_DONE_MARKER = "output/.dubbing_done"


def resolve_windows_binary(name: str) -> str | None:
    """Resolve a binary (e.g. 'ffmpeg' or 'ffprobe') to its real existing path on Windows."""
    found = shutil.which(name)
    if found:
        try:
            resolved = os.path.realpath(found)
            if os.path.isfile(resolved):
                return resolved
        except OSError:
            pass
        if os.path.isfile(found):
            return found
    return None


_LAST_SANITIZED_WIN_PATH: str | None = None


def ensure_windows_ffmpeg_path() -> None:
    """Sanitize Windows PATH (dropping quotes and broken dirs) and hook Popen for bare binaries."""
    global _LAST_SANITIZED_WIN_PATH
    if os.name != "nt":
        return
    cur_path = os.environ.get("PATH", "")
    if cur_path == _LAST_SANITIZED_WIN_PATH and getattr(
        subprocess.Popen.__init__, "_vl_win_patched", False
    ):
        return
    raw_parts = cur_path.split(os.pathsep)
    cleaned_parts: list[str] = []
    seen_lower: set[str] = set()
    for part in raw_parts:
        p = part.strip().strip('"').strip("'")
        if not p:
            continue
        try:
            if not os.path.isdir(p):
                continue
        except OSError:
            continue
        key = p.lower()
        if key not in seen_lower:
            seen_lower.add(key)
            cleaned_parts.append(p)

    user_home = (
        os.environ.get("USERPROFILE")
        or os.environ.get("HOME")
        or os.path.expanduser("~")
    )
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.join(
        user_home, "AppData", "Local"
    )
    exe_dir = os.path.dirname(os.path.realpath(sys.executable))
    prefix_dir = os.path.realpath(sys.prefix)
    candidates: list[str] = [
        os.path.join(exe_dir, "Library", "bin"),
        os.path.join(prefix_dir, "Library", "bin"),
        os.path.join(local_appdata, "Microsoft", "WinGet", "Links"),
        os.path.join(user_home, "scoop", "apps", "ffmpeg", "current", "bin"),
        os.path.join(user_home, "scoop", "shims"),
        r"C:\ProgramData\chocolatey\bin",
        r"C:\ffmpeg\bin",
        r"D:\ffmpeg\bin",
        os.path.join(user_home, ".local", "bin"),
    ]
    winget_pkgs = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_pkgs):
        for ff_exe in glob.glob(
            os.path.join(winget_pkgs, "*FFmpeg*", "*", "bin", "ffmpeg.exe")
        ):
            candidates.append(os.path.dirname(ff_exe))

    for cand in candidates:
        try:
            if not os.path.isdir(cand):
                continue
            resolved_dir = os.path.realpath(cand)
        except OSError:
            resolved_dir = cand
        for d_str in (resolved_dir, cand):
            if os.path.isdir(d_str) and d_str.lower() not in seen_lower:
                seen_lower.add(d_str.lower())
                cleaned_parts.insert(0, d_str)

    os.environ["PATH"] = os.pathsep.join(cleaned_parts)

    # If shutil.which("ffmpeg") resolves through a symlink/shim, also prepend its real target parent
    real_ffmpeg = resolve_windows_binary("ffmpeg")
    if real_ffmpeg:
        real_dir = os.path.dirname(real_ffmpeg)
        if os.path.isdir(real_dir) and real_dir.lower() not in seen_lower:
            os.environ["PATH"] = real_dir + os.pathsep + os.environ["PATH"]

    orig_init = subprocess.Popen.__init__
    if not getattr(orig_init, "_vl_win_patched", False):

        def _patched_popen_init(self, args, *popen_args, **popen_kwargs):
            if (
                popen_kwargs.get("executable") is None
                and not popen_kwargs.get("shell", False)
                and isinstance(args, (list, tuple))
                and args
            ):
                cmd0 = str(args[0])
                if "/" not in cmd0 and "\\" not in cmd0:
                    stem = cmd0.lower().removesuffix(".exe")
                    if stem in {"ffmpeg", "ffprobe"}:
                        resolved_bin = resolve_windows_binary(stem)
                        if resolved_bin:
                            popen_kwargs["executable"] = resolved_bin
            return orig_init(self, args, *popen_args, **popen_kwargs)

        setattr(_patched_popen_init, "_vl_win_patched", True)
        subprocess.Popen.__init__ = _patched_popen_init  # type: ignore[method-assign]
    _LAST_SANITIZED_WIN_PATH = os.environ.get("PATH", "")


ensure_windows_ffmpeg_path()


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
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR",
    "_TEXT_DONE_MARKER",
    "_AUDIO_DONE_MARKER",
    "ensure_windows_ffmpeg_path",
    "resolve_task_path",
    "resolve_windows_binary",
]

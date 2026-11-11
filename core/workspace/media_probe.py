from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import errno
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

_DARWIN_CLONEFILE: Any = None
_DARWIN_CLONEFILE_CHECKED = False


def _try_darwin_clonefile(src: Path, dst: Path) -> bool:
    """Invoke macOS APFS clonefile(2) directly via libc without spawning a subprocess."""
    global _DARWIN_CLONEFILE, _DARWIN_CLONEFILE_CHECKED
    if not _DARWIN_CLONEFILE_CHECKED:
        _DARWIN_CLONEFILE_CHECKED = True
        libc_name = ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib"
        with contextlib.suppress(OSError, AttributeError):
            libc = ctypes.CDLL(libc_name, use_errno=True)
            fn = libc.clonefile
            fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
            fn.restype = ctypes.c_int
            _DARWIN_CLONEFILE = fn
    if _DARWIN_CLONEFILE is None:
        return False
    rc = _DARWIN_CLONEFILE(os.fsencode(src), os.fsencode(dst), 0)
    return rc == 0 and dst.is_file()


def link_or_copy_media(src_path: str | Path, dst_path: str | Path) -> str:
    """Mount a local media file into a task workspace without browser buffering or fragile symlinks.

    Order of preference:
    1. macOS APFS Copy-on-Write clone (`clonefile(2)` / `cp -c`): 0 extra disk blocks, independent inode.
    2. Same-volume hard link (`os.link`): 0 extra disk blocks; survives moving or deleting `src_path`.
    3. Cross-volume physical copy (`shutil.copy2`): used when `src_path` is on another drive (`EXDEV`).
    Never uses `os.symlink`, so moving or deleting the original file never breaks the task.
    """
    src = Path(src_path).resolve()
    dst = Path(dst_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source media file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return "samefile"
        dst.unlink()

    if platform.system() == "Darwin":
        if _try_darwin_clonefile(src, dst):
            return "clonefile"
        res = subprocess.run(
            ["cp", "-c", str(src), str(dst)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode == 0 and dst.is_file():
            return "clonefile"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in (
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
            errno.EINVAL,
        ):
            raise
        shutil.copy2(src, dst)
        return "copy"


def probe_media_metadata(media_path: str | Path) -> dict[str, Any]:
    """Probe display dimensions (accounting for rotation matrix), duration, and audio presence via ffprobe."""
    p = Path(media_path)
    info: dict[str, Any] = {
        "width": 0,
        "height": 0,
        "rotation": 0,
        "display_width": 0,
        "display_height": 0,
        "duration_sec": 0.0,
        "has_audio": False,
        "has_video": False,
        "codec_name": "",
        "audio_codec": "",
        "pix_fmt": "",
    }
    if not p.is_file():
        return info

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,pix_fmt,width,height:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(p),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=15,
        )
        data = json.loads(proc.stdout or "{}")
    except FileNotFoundError:
        return info
    except (subprocess.SubprocessError, OSError, ValueError):
        return info

    fmt_dur = data.get("format", {}).get("duration")
    if fmt_dur is not None:
        with contextlib.suppress(ValueError, TypeError):
            info["duration_sec"] = round(float(fmt_dur), 3)

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            info["has_audio"] = True
            if not info["audio_codec"]:
                info["audio_codec"] = str(stream.get("codec_name") or "").lower()
        elif codec_type == "video" and not info["has_video"]:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            info["has_video"] = True
            info["width"] = width
            info["height"] = height
            info["codec_name"] = str(stream.get("codec_name") or "").lower()
            info["pix_fmt"] = str(stream.get("pix_fmt") or "").lower()
            rot = 0
            tag_rot = (stream.get("tags") or {}).get("rotate")
            if tag_rot is not None:
                with contextlib.suppress(ValueError, TypeError):
                    rot = int(round(float(tag_rot)))
            for sd in stream.get("side_data_list") or []:
                if "rotation" in sd:
                    with contextlib.suppress(ValueError, TypeError):
                        rot = int(round(float(sd["rotation"])))
            rot_norm = abs(rot) % 360
            info["rotation"] = rot
            if rot_norm in (90, 270):
                info["display_width"] = height
                info["display_height"] = width
            else:
                info["display_width"] = width
                info["display_height"] = height
    return info

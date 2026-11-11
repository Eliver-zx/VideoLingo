"""FFmpeg segment decoding, vocal track length matching, and O(search) low-energy window splitting."""

from pathlib import Path
import subprocess

import numpy as np

from core.asr_backend.qwen_asr.config import (
    CUT_SEARCH_SECONDS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
)


def load_audio_segment(path: str | Path, start: float, end: float) -> np.ndarray:
    """Decode [start, end) to 16 kHz mono float32 with the FFmpeg CLI.

    librosa.load deadlocks inside Streamlit's ScriptRunner thread (see whisperX_local).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        str(start),
        "-i",
        str(path),
        "-t",
        str(end - start),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"FFmpeg could not decode {path} ({start}-{end}s): {stderr or f'exit code {proc.returncode}'}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def match_length(wav: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad so the vocal track shares the raw track's sample indices.

    demucs_vl writes stems that start sample-aligned with raw.mp3 (FFmpeg decode and
    encode, so MP3 encoder delay is trimmed); only the decoded tail length can differ.
    """
    if len(wav) >= length:
        return wav[:length]
    return np.concatenate([wav, np.zeros(length - len(wav), dtype=wav.dtype)])


def split_windows(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
    search_seconds: float = CUT_SEARCH_SECONDS,
) -> list[tuple[int, int]]:
    """Split samples into contiguous (start, end) windows of at most window_seconds, cut at low energy."""
    total, max_len = len(wav), int(window_seconds * sr)
    if total <= max_len:
        return [(0, total)]
    search, frame = max(0, int(search_seconds * sr)), max(1, int(0.1 * sr))
    windows: list[tuple[int, int]] = []
    start = 0
    csum = np.empty(max(search, frame) + 1, dtype=np.float64)
    csum[0] = 0.0
    while total - start > max_len:
        right = start + max_len
        left = max(start + 1, right - search)
        seg_len = right - left
        if seg_len >= frame:
            buf = csum[: seg_len + 1]
            np.cumsum(np.abs(wav[left:right]), dtype=np.float64, out=buf[1:])
            energy = buf[frame:] - buf[:-frame]
            cut = left + int(np.argmin(energy)) + frame // 2
        else:
            cut = right
        windows.append((start, cut))
        start = cut
    windows.append((start, total))
    return windows

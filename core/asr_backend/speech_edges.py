# pyright: reportPrivateImportUsage=false
"""Conservative speech boundaries; keep pauses and the original sample timeline."""

from contextlib import contextmanager
import os

SPEECH_PADDING_SECONDS = 1.0


@contextmanager
def _unicode_safe_jit_load(torch_mod):
    """Route torch.jit.load file paths through Python's binary stream so Windows non-ASCII paths never hit C++ ANSI fopen."""
    jit = getattr(torch_mod, "jit", None)
    orig_load = getattr(jit, "load", None)
    if jit is None or orig_load is None:
        yield
        return

    def _stream_load(f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)):
            with open(f, "rb") as fp:
                return orig_load(fp, *args, **kwargs)
        return orig_load(f, *args, **kwargs)

    jit.load = _stream_load
    try:
        yield
    finally:
        jit.load = orig_load


def _speech_regions(audio, sample_rate):
    import torch
    from core.utils import check_cancel, report_progress

    # Importing silero_vad changes PyTorch's process-wide thread count. Restore it
    # before ASR runs, including when loading or inference fails.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        from silero_vad import load_silero_vad, get_speech_timestamps

        report_progress(
            None,
            "VAD 语音边界检测…",
            "正在运行 Silero VAD 检测有效语音边界",
            tag="asr",
            label="VAD",
        )
        with _unicode_safe_jit_load(torch):
            model = load_silero_vad()
        return get_speech_timestamps(
            torch.from_numpy(audio),
            model,
            sampling_rate=sample_rate,
            threshold=0.35,
            min_speech_duration_ms=100,
            speech_pad_ms=0,
            progress_tracking_callback=lambda _: check_cancel(),
        )
    finally:
        torch.set_num_threads(threads)


def trim_speech_windows(audio, windows, sample_rate=16000):
    """Trim only window edges, padding both sides to retain soft/short words.

    A window with no detected speech is left intact: VAD missing quiet speech
    must not silently discard it. Internal pauses are never removed. Returned
    indices still refer to the original audio, for both ASR and alignment.
    """
    if not len(audio) or not windows:
        return windows
    regions = _speech_regions(audio, sample_rate)
    padding = round(SPEECH_PADDING_SECONDS * sample_rate)
    trimmed = []
    for start, end in windows:
        speech = [r for r in regions if r["end"] > start and r["start"] < end]
        if not speech:
            trimmed.append((start, end))
            continue
        trimmed.append(
            (
                max(start, speech[0]["start"] - padding),
                min(end, speech[-1]["end"] + padding),
            )
        )
    return trimmed

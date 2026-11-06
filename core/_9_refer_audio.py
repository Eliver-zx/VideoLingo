import os
from importlib.util import find_spec
from core.utils import log_event, progress_scope, report_progress
from core.utils.models import (
    _AUDIO_REFERS_DIR,
    _AUDIO_SEGS_DIR,  # noqa: F401
    _BACKGROUND_AUDIO_FILE,
    _RAW_AUDIO_FILE,
    _VOCAL_AUDIO_FILE,
    resolve_task_path,
)
from core.workspace import get_active_workspace
import pandas as pd
import soundfile as sf


def time_to_samples(time_str, sr):
    """Unified time conversion function"""
    h, m, s = time_str.split(":")
    s, ms = s.split(",") if "," in s else (s, "0")
    seconds = int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000
    return int(seconds * sr)


def extract_audio(audio_data, sr, start_time, end_time, out_file):
    """Simplified audio extraction function"""
    start = time_to_samples(start_time, sr)
    end = time_to_samples(end_time, sr)
    sf.write(out_file, audio_data[start:end], sr)


def _ensure_high_quality_raw_audio(raw_audio_file: str) -> None:
    """Upgrade a lightweight 16 kHz mono ASR raw.mp3 to stereo high-quality audio when dubbing runs later."""
    if not os.path.exists(raw_audio_file):
        return
    try:
        info = sf.info(raw_audio_file)
    except (OSError, RuntimeError):
        return
    if info.samplerate >= 32000 and info.channels >= 2:
        return
    from core._1_ytdlp import find_media_file
    from core.asr_backend.audio_preprocess import (
        convert_video_to_audio,
        prepare_audio_for_asr,
    )

    try:
        media_file, media_type = find_media_file()
    except ValueError:
        return
    if media_type == "video":
        convert_video_to_audio(media_file, high_quality=True, force=True)
    elif media_type == "audio":
        prepare_audio_for_asr(media_file, high_quality=True, force=True)
    else:
        return
    for stale_stem in (_VOCAL_AUDIO_FILE, _BACKGROUND_AUDIO_FILE):
        stale_path = resolve_task_path(stale_stem)
        if os.path.exists(stale_path):
            os.remove(stale_path)


def extract_refer_audio_main():
    raw_audio_file = resolve_task_path(_RAW_AUDIO_FILE)
    _ensure_high_quality_raw_audio(raw_audio_file)
    vocal_audio_file = resolve_task_path(_VOCAL_AUDIO_FILE)
    refers_dir = resolve_task_path(_AUDIO_REFERS_DIR)
    reference_audio = raw_audio_file
    try:
        from core.asr_backend.demucs_vl import demucs_audio, separation_backend_package

        pkg = separation_backend_package()
        if find_spec(pkg) is not None:
            with progress_scope(0.05, 0.70):
                demucs_audio()  # Dubbing normally keeps the original background track.
            reference_audio = vocal_audio_file
        elif pkg != "demucs" and find_spec("demucs") is not None:
            with progress_scope(0.05, 0.70):
                demucs_audio(model_key="htdemucs")
            reference_audio = vocal_audio_file
        else:
            log_event(
                "人声分离组件不可用，使用原始音频提取参考音且不保留背景音轨",
                tag="warn",
                label="VOCAL",
                level="warn",
            )
    except ImportError:
        log_event(
            "人声分离组件不可用，使用原始音频提取参考音且不保留背景音轨",
            tag="warn",
            label="VOCAL",
            level="warn",
        )
    first_refer = os.path.join(refers_dir, "1.wav")
    if os.path.exists(first_refer):
        try:
            ref_up_to_date = not os.path.exists(reference_audio) or os.path.getmtime(
                first_refer
            ) >= os.path.getmtime(reference_audio)
        except OSError:
            ref_up_to_date = True
        if ref_up_to_date:
            report_progress(
                1.0,
                "复用已有参考音频切片",
                "参考音频切片已存在且为最新，跳过提取",
                tag="asr",
                label="CACHE",
            )
            return

    # Create output directory
    os.makedirs(refers_dir, exist_ok=True)

    ws = get_active_workspace()
    df = ws.load_dub_tasks_df()
    data, sr = sf.read(reference_audio)

    total_rows = max(1, len(df))
    report_progress(
        0.72,
        f"提取参考音频 0/{len(df)}",
        f"开始切分克隆参考音频（共 {len(df)} 段）",
        tag="asr",
        label="TTS",
    )
    for row_idx, (_, row) in enumerate(df.iterrows(), 1):
        out_file = os.path.join(refers_dir, f"{row['number']}.wav")
        start_time = row["start_time"]
        end_time = row["end_time"]
        raw_start_val = row.get("raw_start_time")
        raw_end_val = row.get("raw_end_time")
        if (
            raw_start_val is not None
            and raw_end_val is not None
            and not pd.isna(raw_start_val)
            and not pd.isna(raw_end_val)
        ):
            raw_start = time_to_samples(str(raw_start_val), sr)
            raw_end = time_to_samples(str(raw_end_val), sr)
            if (raw_end - raw_start) >= int(1.0 * sr):
                start_time = str(raw_start_val)
                end_time = str(raw_end_val)
        extract_audio(data, sr, start_time, end_time, out_file)
        if row_idx == len(df) or row_idx % 20 == 0:
            report_progress(
                0.72 + 0.28 * (row_idx / total_rows),
                f"提取参考音频 {row_idx}/{len(df)}",
                tag="asr",
                label="TTS",
            )

    report_progress(
        1.0,
        f"已提取 {len(df)} 段参考音频",
        f"参考音频切片提取完成（共 {len(df)} 段）",
        tag="asr",
        label="TTS",
    )


if __name__ == "__main__":
    extract_refer_audio_main()

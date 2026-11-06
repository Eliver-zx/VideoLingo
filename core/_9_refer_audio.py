import os
from importlib.util import find_spec
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from core.utils import rprint
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

console = Console()


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
            demucs_audio()  # Dubbing normally keeps the original background track.
            reference_audio = vocal_audio_file
        elif pkg != "demucs" and find_spec("demucs") is not None:
            demucs_audio(model_key="htdemucs")
            reference_audio = vocal_audio_file
        else:
            rprint(
                Panel(
                    "Vocal separation is unavailable; using the original audio for references. "
                    "The dubbed video will not keep the original background sound.",
                    title="Info",
                    border_style="yellow",
                )
            )
    except ImportError:
        rprint(
            Panel(
                "Vocal separation is unavailable; using the original audio for references. "
                "The dubbed video will not keep the original background sound.",
                title="Info",
                border_style="yellow",
            )
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
            rprint(
                Panel(
                    "Reference audio already exists, skipping extraction",
                    title="Info",
                    border_style="blue",
                )
            )
            return

    # Create output directory
    os.makedirs(refers_dir, exist_ok=True)

    ws = get_active_workspace()
    df = ws.load_dub_tasks_df()
    data, sr = sf.read(reference_audio)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Extracting audio segments...", total=len(df))

        for _, row in df.iterrows():
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
            progress.update(task, advance=1)

    rprint(
        Panel(
            f"Audio segments saved to {refers_dir}",
            title="Success",
            border_style="green",
        )
    )


if __name__ == "__main__":
    extract_refer_audio_main()

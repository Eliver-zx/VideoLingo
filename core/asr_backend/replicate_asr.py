import io
import time
from typing import Any, Dict, Optional

from rich import print as rprint

from core.asr_backend.audio_preprocess import audio_slice_wav
from core.utils import check_cancel, load_key_or, update_key

REPLICATE_PARAKEET_MODEL = (
    "huanshere/parakeet-tdt-0.6b-v2:6f0fb1d0715174c31e76601f8b98f5e2d7a616cd7a0bf00584c2c820ffb039a4"
)
REPLICATE_WHISPERX_MODEL = (
    "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
)
SAMPLE_RATE = 16000


def configured_credentials() -> str:
    api_key = str(load_key_or("whisper.replicate_api_key", "") or "").strip()
    if not api_key or api_key.lower().startswith(("your_", "your ")):
        raise ValueError("Set whisper.replicate_api_key before using Replicate ASR")
    return api_key


def _get_replicate_client(api_token: str) -> Any:
    from importlib import import_module
    try:
        replicate = import_module("replicate")
    except ImportError as exc:
        raise RuntimeError("The 'replicate' package is required for Replicate ASR (`pip install replicate`).") from exc
    return replicate.Client(api_token=api_token)


def normalize_replicate_output(output: Dict[str, Any], offset: float = 0.0, fallback_lang: str = "en") -> Dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError(f"Unexpected Replicate ASR output type: {type(output).__name__}")

    raw_segments = output.get("segments")
    if "word_timestamps" in output and not raw_segments:
        raw_segments = [{"words": output["word_timestamps"]}]
    if not isinstance(raw_segments, list):
        raw_segments = []

    segments = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        words = []
        for w in (seg.get("words") or []):
            if not isinstance(w, dict):
                continue
            word_text = str(w.get("word") or w.get("text") or "").strip()
            if not word_text:
                continue
            w_item: Dict[str, Any] = {"word": word_text}
            if isinstance(w.get("start"), (int, float)):
                w_item["start"] = round(float(w["start"]) + offset, 3)
            if isinstance(w.get("end"), (int, float)):
                w_item["end"] = round(float(w["end"]) + offset, 3)
            words.append(w_item)

        seg_text = str(seg.get("text") or " ".join(w["word"] for w in words)).strip()
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        if isinstance(seg_start, (int, float)):
            seg_start = round(float(seg_start) + offset, 3)
        elif words and "start" in words[0]:
            seg_start = words[0]["start"]

        if isinstance(seg_end, (int, float)):
            seg_end = round(float(seg_end) + offset, 3)
        elif words and "end" in words[-1]:
            seg_end = words[-1]["end"]

        if words or seg_text:
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "words": words,
            })

    detected_lang = output.get("detected_language") or output.get("language") or fallback_lang
    if not detected_lang or str(detected_lang).lower() == "auto":
        detected_lang = "en"

    return {
        "segments": segments,
        "language": str(detected_lang),
    }


def transcribe_audio_replicate_parakeet(
    raw_audio_path: str,
    vocal_audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, Any]:
    check_cancel()
    api_token = configured_credentials()
    language = str(load_key_or("whisper.language", "en") or "en").strip()
    audio_bytes = audio_slice_wav(vocal_audio_path, start, end, sample_rate=SAMPLE_RATE)
    offset = float(start) if start is not None else 0.0

    client = _get_replicate_client(api_token)
    input_params = {
        "audio": io.BytesIO(audio_bytes),
        "max_duration_minutes": 0,
        "max_speakers": -1,
        "min_speakers": -1,
        "return_word_timestamps": True,
        "speaker_diarization": False,
    }
    rprint("[cyan]🎤 Transcribing audio with Replicate Parakeet-TDT-0.6b-v2...[/cyan]")
    t0 = time.perf_counter()
    check_cancel()
    output = client.run(REPLICATE_PARAKEET_MODEL, input=input_params)
    parsed = normalize_replicate_output(output, offset=offset, fallback_lang=language)
    if parsed.get("language"):
        update_key("whisper.detected_language", parsed["language"])
    rprint(f"[green]✓ Replicate Parakeet transcription completed in {time.perf_counter() - t0:.2f}s[/green]")
    return parsed


def transcribe_audio_replicate_whisper(
    raw_audio_path: str,
    vocal_audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, Any]:
    check_cancel()
    api_token = configured_credentials()
    language = str(load_key_or("whisper.language", "en") or "en").strip()
    audio_bytes = audio_slice_wav(vocal_audio_path, start, end, sample_rate=SAMPLE_RATE)
    offset = float(start) if start is not None else 0.0

    client = _get_replicate_client(api_token)
    input_params: Dict[str, Any] = {
        "audio_file": io.BytesIO(audio_bytes),
        "debug": False,
        "vad_onset": 0.5,
        "batch_size": 64,
        "vad_offset": 0.363,
        "diarization": False,
        "temperature": 0,
        "align_output": True,
        "language_detection_min_prob": 0,
        "language_detection_max_tries": 5,
    }
    if language and language.lower() != "auto":
        input_params["language"] = language

    rprint("[cyan]🎤 Transcribing audio with Replicate WhisperX...[/cyan]")
    t0 = time.perf_counter()
    check_cancel()
    output = client.run(REPLICATE_WHISPERX_MODEL, input=input_params)
    parsed = normalize_replicate_output(output, offset=offset, fallback_lang=language)
    if parsed.get("language"):
        update_key("whisper.detected_language", parsed["language"])
    rprint(f"[green]✓ Replicate WhisperX transcription completed in {time.perf_counter() - t0:.2f}s[/green]")
    return parsed

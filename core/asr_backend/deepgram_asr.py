import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from rich import print as rprint

from core.asr_backend.audio_preprocess import audio_slice_wav
from core.utils import check_cancel, load_key_or, update_key

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"
SAMPLE_RATE = 16000


def configured_credentials() -> Tuple[str, str]:
    api_key = str(load_key_or("whisper.deepgram_api_key", "") or "").strip()
    if not api_key or api_key.lower().startswith(("your_", "your ")):
        raise ValueError("Set whisper.deepgram_api_key before using Deepgram ASR")
    model = str(load_key_or("whisper.deepgram_model", "nova-3") or "nova-3").strip()
    return api_key, model


def _adjust_time(ts: float, offset: float) -> float:
    return round(float(ts) + offset, 3) if offset else round(float(ts), 3)


def deepgram2whisper(dg_json: Dict[str, Any], offset: float = 0.0, fallback_lang: str = "en") -> Dict[str, Any]:
    results = dg_json.get("results") or {}
    channels = results.get("channels") or []
    alt = (channels[0].get("alternatives") or [{}])[0] if channels else {}
    detected_lang = (
        (channels[0].get("detected_language") if channels else None)
        or results.get("language")
        or fallback_lang
    )
    if not detected_lang or detected_lang == "auto":
        detected_lang = "en"

    words: List[dict] = alt.get("words") or []
    utterances: List[dict] = results.get("utterances") or []
    segments: List[dict] = []

    if utterances:
        for utt in utterances:
            utt_start = float(utt.get("start", 0.0))
            utt_end = float(utt.get("end", utt_start))
            segment_words = [
                {
                    "word": str(w.get("punctuated_word") or w.get("word") or "").strip(),
                    "start": _adjust_time(w["start"], offset),
                    "end": _adjust_time(w["end"], offset),
                }
                for w in words
                if "start" in w and "end" in w
                and float(w["start"]) >= utt_start - 1e-3
                and float(w["end"]) <= utt_end + 1e-3
                and str(w.get("punctuated_word") or w.get("word") or "").strip()
            ]
            segments.append({
                "start": _adjust_time(utt_start, offset),
                "end": _adjust_time(utt_end, offset),
                "text": str(utt.get("transcript") or "").strip(),
                "words": segment_words,
            })
    elif words:
        valid_words = [
            {
                "word": str(w.get("punctuated_word") or w.get("word") or "").strip(),
                "start": _adjust_time(w["start"], offset),
                "end": _adjust_time(w["end"], offset),
            }
            for w in words
            if "start" in w and "end" in w and str(w.get("punctuated_word") or w.get("word") or "").strip()
        ]
        if valid_words:
            segments.append({
                "start": valid_words[0]["start"],
                "end": valid_words[-1]["end"],
                "text": str(alt.get("transcript") or " ".join(w["word"] for w in valid_words)).strip(),
                "words": valid_words,
            })

    return {
        "segments": segments,
        "language": str(detected_lang),
    }


def transcribe_audio_deepgram(
    raw_audio_path: str,
    vocal_audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, Any]:
    check_cancel()
    api_key, model = configured_credentials()
    language = str(load_key_or("whisper.language", "en") or "en").strip()

    audio_bytes = audio_slice_wav(vocal_audio_path, start, end, sample_rate=SAMPLE_RATE)
    offset = float(start) if start is not None else 0.0

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    params: Dict[str, str] = {
        "model": model,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    if language and language.lower() != "auto":
        params["language"] = language
    else:
        params["detect_language"] = "true"

    rprint(f"[cyan]🎤 Transcribing audio with Deepgram (model={model}, language={language})...[/cyan]")
    start_t = time.perf_counter()
    check_cancel()
    response = requests.post(
        DEEPGRAM_URL,
        headers=headers,
        params=params,
        data=audio_bytes,
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Deepgram ASR failed (HTTP {response.status_code}): {response.text[:300]}")
    dg_json = response.json()

    elapsed = time.perf_counter() - start_t
    rprint(f"[green]✓ Deepgram transcription completed in {elapsed:.2f}s[/green]")

    parsed = deepgram2whisper(dg_json, offset=offset, fallback_lang=language)
    if parsed.get("language"):
        update_key("whisper.detected_language", parsed["language"])
    return parsed

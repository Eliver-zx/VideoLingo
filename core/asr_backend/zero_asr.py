import time
from typing import Any, Dict, Optional

import requests
from rich import print as rprint

from core.asr_backend.audio_preprocess import audio_slice_wav
from core.utils import check_cancel, load_key_or, update_key

UPLOAD_URL = "https://api.gladia.io/v2/upload/"
TRANSCRIPTION_URL = "https://api.gladia.io/v2/transcription/"
SAMPLE_RATE = 16000


def configured_credentials() -> str:
    api_key = str(load_key_or("whisper.whisperzero_api_key", "") or "").strip()
    if not api_key or api_key.lower().startswith(("your_", "your ")):
        raise ValueError("Set whisper.whisperzero_api_key before using Gladia Zero ASR")
    return api_key


def zero2whisper(result: Dict[str, Any], offset: float = 0.0, fallback_lang: str = "en") -> Dict[str, Any]:
    transcription = result.get("transcription") or {}
    utterances = transcription.get("utterances") or []
    languages = transcription.get("languages") or []
    detected_lang = (languages[0] if languages else None) or result.get("language") or fallback_lang
    if not detected_lang or detected_lang == "auto":
        detected_lang = "en"

    segments = []
    for u in utterances:
        words = [
            {
                "word": str(w.get("word") or "").strip(),
                "start": round(float(w["start"]) + offset, 3),
                "end": round(float(w["end"]) + offset, 3),
            }
            for w in (u.get("words") or [])
            if "start" in w and "end" in w and str(w.get("word") or "").strip()
        ]
        if "start" in u and "end" in u:
            segments.append({
                "start": round(float(u["start"]) + offset, 3),
                "end": round(float(u["end"]) + offset, 3),
                "text": str(u.get("text") or "").strip(),
                "words": words,
            })

    return {
        "segments": segments,
        "language": str(detected_lang),
    }


def transcribe_audio_zero(
    raw_audio_path: str,
    vocal_audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, Any]:
    check_cancel()
    api_key = configured_credentials()
    language = str(load_key_or("whisper.language", "en") or "en").strip()

    audio_bytes = audio_slice_wav(vocal_audio_path, start, end, sample_rate=SAMPLE_RATE)
    offset = float(start) if start is not None else 0.0

    headers: Dict[str, str] = {
        "x-gladia-key": api_key,
        "accept": "application/json",
    }
    files = [("audio", ("slice.wav", audio_bytes, "audio/wav"))]

    start_time = time.perf_counter()
    rprint("[cyan]🎤 Uploading audio slice to Gladia Zero...[/cyan]")
    upload_resp = requests.post(UPLOAD_URL, headers=headers, files=files, timeout=120)
    if upload_resp.status_code >= 400:
        raise RuntimeError(f"Gladia upload failed (HTTP {upload_resp.status_code}): {upload_resp.text[:300]}")
    audio_url = upload_resp.json().get("audio_url")
    if not audio_url:
        raise RuntimeError("Gladia upload response did not include audio_url")

    check_cancel()
    headers["Content-Type"] = "application/json"
    payload: Dict[str, Any] = {
        "audio_url": audio_url,
        "diarization": False,
    }
    if language and language.lower() != "auto":
        payload["detect_language"] = False
        payload["language"] = language
    else:
        payload["detect_language"] = True

    transcribe_resp = requests.post(TRANSCRIPTION_URL, headers=headers, json=payload, timeout=60)
    if transcribe_resp.status_code >= 400:
        raise RuntimeError(f"Gladia transcription request failed (HTTP {transcribe_resp.status_code}): {transcribe_resp.text[:300]}")
    result_url = transcribe_resp.json().get("result_url")
    if not result_url:
        raise RuntimeError("Gladia transcription response did not include result_url")

    while True:
        check_cancel()
        poll_resp = requests.get(result_url, headers=headers, timeout=30)
        if poll_resp.status_code >= 400:
            raise RuntimeError(f"Gladia poll failed (HTTP {poll_resp.status_code}): {poll_resp.text[:300]}")
        poll_json = poll_resp.json()
        status = poll_json.get("status")
        if status == "done":
            result = poll_json.get("result") or {}
            break
        if status == "error":
            error_msg = poll_json.get("error_code") or poll_json.get("message") or "Gladia returned error status"
            raise RuntimeError(f"Gladia transcription failed: {error_msg}")
        time.sleep(2)

    parsed = zero2whisper(result, offset=offset, fallback_lang=language)
    if parsed.get("language"):
        update_key("whisper.detected_language", parsed["language"])
    rprint(f"[green]✓ Gladia Zero transcription completed in {time.perf_counter() - start_time:.2f}s[/green]")
    return parsed

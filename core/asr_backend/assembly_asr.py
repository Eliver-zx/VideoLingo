import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.asr_backend.audio_preprocess import audio_slice_wav
from core.utils import (
    check_cancel,
    load_key_or,
    report_progress,
    set_runtime_state as update_key,
)

UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
SAMPLE_RATE = 16000


def configured_credentials() -> Tuple[str, str, str]:
    api_key = str(load_key_or("whisper.assembly_api_key", "") or "").strip()
    if not api_key or api_key.lower().startswith(("your_", "your ")):
        raise ValueError("Set whisper.assembly_api_key before using AssemblyAI ASR")
    lang = str(
        load_key_or("whisper.assemblyai_lang", "")
        or load_key_or("whisper.language", "en")
        or "en"
    ).strip()
    model = str(
        load_key_or("whisper.assemblyai_model", "universal") or "universal"
    ).strip()
    return api_key, lang, model


def convert_assemblyai_response(
    response_data: Dict[str, Any],
    start_time_offset: Optional[float] = None,
    fallback_lang: str = "en",
) -> Dict[str, Any]:
    segments: List[dict] = []
    current_words: List[dict] = []
    offset = float(start_time_offset) if start_time_offset is not None else 0.0

    for word in response_data.get("words") or []:
        text = str(word.get("text") or "").strip()
        if not text or "start" not in word or "end" not in word:
            continue
        start = round(int(word["start"]) / 1000.0 + offset, 3)
        end = round(int(word["end"]) / 1000.0 + offset, 3)
        current_words.append({"word": text, "start": start, "end": end})

        if text[-1] in ".!?。！？":
            segments.append(
                {
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "text": " ".join(w["word"] for w in current_words),
                    "words": current_words,
                }
            )
            current_words = []

    if current_words:
        segments.append(
            {
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": " ".join(w["word"] for w in current_words),
                "words": current_words,
            }
        )

    detected_lang = response_data.get("language_code") or fallback_lang
    if not detected_lang or detected_lang == "auto":
        detected_lang = "en"

    return {
        "segments": segments,
        "language": str(detected_lang),
    }


def transcribe_audio_assembly(
    raw_audio_path: str,
    vocal_audio_path: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> Dict[str, Any]:
    check_cancel()
    api_key, lang, model = configured_credentials()
    audio_bytes = audio_slice_wav(vocal_audio_path, start, end, sample_rate=SAMPLE_RATE)

    headers: Dict[str, str] = {"Authorization": api_key}
    start_time = time.perf_counter()

    report_progress(
        0.15,
        "上传至 AssemblyAI…",
        "正在上传音频切片至 AssemblyAI",
        tag="asr",
        label="ASR",
    )
    upload_resp = requests.post(
        UPLOAD_URL, headers=headers, data=audio_bytes, timeout=180
    )
    if upload_resp.status_code >= 400:
        raise RuntimeError(
            f"AssemblyAI upload failed (HTTP {upload_resp.status_code}): {upload_resp.text[:300]}"
        )
    audio_url = upload_resp.json().get("upload_url")
    if not audio_url:
        raise RuntimeError("AssemblyAI upload response did not include upload_url")

    check_cancel()
    headers["Content-Type"] = "application/json"
    transcript_params: Dict[str, Any] = {
        "audio_url": audio_url,
        "speech_model": model,
    }
    if lang and lang.lower() != "auto":
        transcript_params["language_code"] = lang
    else:
        transcript_params["language_detection"] = True

    submit_resp = requests.post(
        TRANSCRIPT_URL, headers=headers, json=transcript_params, timeout=60
    )
    if submit_resp.status_code >= 400:
        raise RuntimeError(
            f"AssemblyAI submit failed (HTTP {submit_resp.status_code}): {submit_resp.text[:300]}"
        )
    transcript_id = submit_resp.json().get("id")
    if not transcript_id:
        raise RuntimeError("AssemblyAI transcript response did not include id")

    report_progress(
        0.35,
        "等待 AssemblyAI 识别…",
        f"AssemblyAI 任务已提交 (id={transcript_id})，正在等待识别结果",
        tag="asr",
        label="ASR",
    )
    poll_url = f"{TRANSCRIPT_URL}/{transcript_id}"
    poll_ticks = 0
    while True:
        check_cancel()
        poll_resp = requests.get(poll_url, headers=headers, timeout=30)
        if poll_resp.status_code >= 400:
            raise RuntimeError(
                f"AssemblyAI poll failed (HTTP {poll_resp.status_code}): {poll_resp.text[:300]}"
            )
        raw_response = poll_resp.json()
        status = raw_response.get("status")
        if status == "completed":
            break
        if status == "error":
            error_msg = (
                raw_response.get("error") or "Unknown AssemblyAI transcription error"
            )
            raise RuntimeError(f"AssemblyAI transcription error: {error_msg}")
        poll_ticks += 1
        report_progress(
            min(0.92, 0.35 + poll_ticks * 0.05),
            f"AssemblyAI {status or 'processing'}…",
            tag="asr",
            label="ASR",
        )
        time.sleep(3)

    parsed = convert_assemblyai_response(
        raw_response, start_time_offset=start, fallback_lang=lang
    )
    if parsed.get("language"):
        update_key("whisper.detected_language", parsed["language"])
    report_progress(
        0.98,
        "AssemblyAI 识别完成",
        f"AssemblyAI 语音识别完成 (耗时 {time.perf_counter() - start_time:.2f}s)",
        tag="asr",
        label="ASR",
    )
    return parsed

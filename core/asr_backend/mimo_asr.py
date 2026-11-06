# pyright: reportPrivateImportUsage=false
import base64
from collections import Counter
import io
import time
from typing import Any
import wave

import numpy as np
import requests

from core.asr_backend import qwen_asr_local as qwen
from core.asr_backend.speech_edges import trim_speech_windows
from core.utils import (
    check_cancel,
    except_handler,
    load_key_or,
    log_event,
    report_progress,
    set_runtime_state as update_key,
)

MIMO_API_URL = "https://api.xiaomimimo.com/v1/chat/completions"
DEFAULT_MIMO_MODEL = "mimo-v2.5-asr"
SAMPLE_RATE = qwen.SAMPLE_RATE
ATTEMPTS = 3
RETRY_STATUS = {408, 429}
REQUEST_TIMEOUT = 180

_SCRIPT_TO_QWEN_LANG = {
    "Han": "Chinese",
    "Kana": "Japanese",
    "Hangul": "Korean",
    "Cyrillic": "Russian",
    "Arabic": "Arabic",
    "Greek": "Greek",
    "Devanagari": "Hindi",
    "Thai": "Thai",
    "Latin": "English",
}


def configured_credentials() -> tuple[str, str]:
    api_key = str(load_key_or("whisper.mimo_api_key", "") or "").strip()
    if not api_key or api_key.lower().startswith(("your_", "your ")):
        raise ValueError("Set whisper.mimo_api_key before using MiMo ASR")
    model = str(
        load_key_or("whisper.mimo_model", DEFAULT_MIMO_MODEL) or DEFAULT_MIMO_MODEL
    ).strip()
    if not model:
        model = DEFAULT_MIMO_MODEL
    return api_key, model


def mimo_language_option(language: str | None) -> str:
    lang = str(language or "auto").strip().lower()
    if lang in ("zh", "yue", "zh-cn", "zh-tw", "zh-hk"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    return "auto"


def wav_bytes_from_float32(clip: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(clip, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def extract_mimo_text(resp_json: dict[str, Any]) -> str:
    if not isinstance(resp_json, dict):
        raise ValueError(
            f"Unexpected MiMo ASR response type: {type(resp_json).__name__}"
        )
    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("MiMo ASR response did not contain choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    raw_msg = first.get("message")
    message = raw_msg if isinstance(raw_msg, dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            str(part.get("text") or "") for part in content if isinstance(part, dict)
        ]
        return "".join(parts).strip()
    return str(content or "").strip()


def infer_qwen_language_from_text(
    text: str, configured_language: str | None = None
) -> str:
    configured = str(configured_language or "auto").strip().lower()
    if configured and configured != "auto" and configured in qwen.ISO_TO_QWEN:
        return qwen.ISO_TO_QWEN[configured]
    counts = qwen.script_counts(text)
    if not counts:
        return "English"
    cjk = counts.get("Han", 0) + counts.get("Kana", 0)
    if cjk > 0 and counts.get("Kana", 0) / cjk >= qwen.JA_MIN_KANA_SHARE:
        return "Japanese"
    weighted = {
        script: n * (3 if script in ("Han", "Kana", "Hangul") else 1)
        for script, n in counts.items()
        if script in _SCRIPT_TO_QWEN_LANG
    }
    if not weighted:
        return "English"
    dominant = max(weighted, key=lambda k: weighted[k])
    return _SCRIPT_TO_QWEN_LANG.get(dominant, "English")


def request_mimo_transcription(
    clip: np.ndarray,
    api_key: str,
    model: str,
    language: str | None,
) -> str:
    wav_bytes = wav_bytes_from_float32(clip, sample_rate=SAMPLE_RATE)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:audio/wav;base64,{audio_b64}",
                            "format": "wav",
                        },
                    }
                ],
            }
        ],
        "asr_options": {
            "language": mimo_language_option(language),
        },
    }
    headers = {
        "api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = None
    for attempt in range(1, ATTEMPTS + 1):
        check_cancel()
        try:
            response = requests.post(
                MIMO_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == ATTEMPTS:
                raise
            log_event(
                f"MiMo ASR 连接失败 ({error})，正在重试 {attempt}/{ATTEMPTS - 1}",
                tag="warn",
                label="RETRY",
                level="warn",
            )
        else:
            retryable = (
                response.status_code in RETRY_STATUS
                or 500 <= response.status_code < 600
            )
            if response.ok or not retryable or attempt == ATTEMPTS:
                break
            log_event(
                f"MiMo ASR HTTP {response.status_code}，正在重试 {attempt}/{ATTEMPTS - 1}",
                tag="warn",
                label="RETRY",
                level="warn",
            )
        time.sleep(3 * attempt)

    assert response is not None
    if not response.ok:
        detail = response.text[:500].replace(api_key, "[REDACTED]")
        raise RuntimeError(
            f"MiMo ASR request failed (HTTP {response.status_code}): {detail}"
        )
    return extract_mimo_text(response.json())


def transcribe_window_mimo(
    raw: np.ndarray,
    a: int,
    b: int,
    api_key: str,
    model: str,
    configured_language: str | None,
    offset: float = 0.0,
) -> list[tuple[int, int, str, str]]:
    text = request_mimo_transcription(raw[a:b], api_key, model, configured_language)
    if not qwen.is_repetitive(text):
        qwen_lang = infer_qwen_language_from_text(text, configured_language)
        return [(a, b, qwen_lang, text)]

    span = f"{offset + a / SAMPLE_RATE:.1f}-{offset + b / SAMPLE_RATE:.1f}s"
    log_event(
        f"MiMo ASR 窗口 {span} 输出复读退化，拆分为 {qwen.RETRY_WINDOW_SECONDS}s 子窗口重试",
        tag="warn",
        label="ASR",
        level="warn",
    )
    pieces: list[tuple[int, int, str, str]] = []
    parts = qwen.split_windows(
        raw[a:b],
        window_seconds=qwen.RETRY_WINDOW_SECONDS,
        search_seconds=qwen.RETRY_SEARCH_SECONDS,
    )
    for sa, sb in parts:
        if sb - sa >= int(qwen.MIN_WINDOW_SECONDS * SAMPLE_RATE):
            sub_text = request_mimo_transcription(
                raw[a + sa : a + sb], api_key, model, configured_language
            )
            sub_lang = infer_qwen_language_from_text(sub_text, configured_language)
            pieces.append((a + sa, a + sb, sub_lang, sub_text))
    return pieces


@except_handler("MiMo ASR processing error:")
def transcribe_audio_mimo(raw_audio_file, vocal_audio_file, start=None, end=None):
    check_cancel()
    api_key, model = configured_credentials()
    engine = qwen.resolve_engine()
    aligner_repo = qwen.MODELS[engine]["aligner"]
    configured = str(load_key_or("whisper.language", "auto") or "auto").strip()
    s_offset = float(start) if start is not None else 0.0
    e_offset = float(end) if end is not None else s_offset + float(qwen.WINDOW_SECONDS)

    report_progress(
        0.02,
        f"准备片段 {s_offset:.0f}s–{e_offset:.0f}s",
        f"准备 MiMo ASR ({model}) + ForcedAligner ({engine}) 识别区间 {s_offset:.1f}s–{e_offset:.1f}s",
        tag="asr",
        label="ASR",
    )

    raw = qwen.load_audio_segment(raw_audio_file, s_offset, e_offset)
    if vocal_audio_file == raw_audio_file:
        vocal = raw
    else:
        vocal = qwen.match_length(
            qwen.load_audio_segment(vocal_audio_file, s_offset, e_offset), len(raw)
        )

    windows = [
        (a, b)
        for a, b in qwen.split_windows(raw)
        if b - a >= int(qwen.MIN_WINDOW_SECONDS * SAMPLE_RATE)
    ]
    windows = trim_speech_windows(raw, windows)

    t0 = time.time()
    pieces: list[tuple[int, int, str, str]] = []
    total_w = max(1, len(windows))
    for w_idx, (a, b) in enumerate(windows, 1):
        check_cancel()
        span_label = (
            f"{s_offset + a / SAMPLE_RATE:.1f}s–{s_offset + b / SAMPLE_RATE:.1f}s"
        )
        report_progress(
            0.10 + 0.60 * ((w_idx - 1) / total_w),
            f"请求 MiMo {w_idx}/{total_w} ({span_label})",
            f"🎙️ [{w_idx}/{total_w}] 正在请求 MiMo ({model}) 识别窗口 {span_label}...",
            tag="asr",
            label="ASR",
        )
        t_win = time.time()
        window_pieces = transcribe_window_mimo(
            raw, a, b, api_key, model, configured, offset=s_offset
        )
        win_text = " ".join(text for *_, text in window_pieces).strip()
        preview = (
            (win_text[:48] + "…") if len(win_text) > 48 else (win_text or "<静音>")
        )
        report_progress(
            0.10 + 0.60 * (w_idx / total_w),
            f"已完成窗口 {w_idx}/{total_w}",
            f"✅ [{w_idx}/{total_w}] 完成 MiMo 窗口 {span_label} ({time.time() - t_win:.1f}s, {len(win_text)} 字符): {preview}",
            tag="asr",
            label="ASR",
        )
        pieces.extend(window_pieces)

    log_event(
        f"MiMo ({model}) 文本识别耗时 {time.time() - t0:.2f}s",
        tag="asr",
        label="ASR",
    )

    if configured and configured.lower() != "auto":
        detected = configured
    else:
        votes: Counter[str] = Counter()
        for a, b, q_lang, text in pieces:
            iso = qwen.iso_language(q_lang)
            if text and iso:
                votes[iso] += b - a
        detected = max(votes, key=lambda k: votes[k]) if votes else "en"

    update_key("whisper.detected_language", detected)

    t_align = time.time()
    default_qwen_lang = qwen.ISO_TO_QWEN.get(detected or "") or "English"
    todo = [
        (a, b, text, (qwen.primary_language(q_lang) or default_qwen_lang))
        for a, b, q_lang, text in pieces
        if text
    ]
    jobs = [(vocal[a:b], text, q_lang) for a, b, text, q_lang in todo]
    aligned = qwen._align(engine, aligner_repo, jobs) if jobs else []
    log_event(
        f"ForcedAligner 词级对齐耗时 {time.time() - t_align:.2f}s",
        tag="asr",
        label="ASR",
    )

    segments = []
    for (a, b, text, _), items in zip(todo, aligned):
        words = qwen.attach_words(
            text, items, s_offset + a / SAMPLE_RATE, limit=(b - a) / SAMPLE_RATE
        )
        if words:
            segments.append(
                {
                    "text": text,
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "words": words,
                }
            )
    report_progress(
        0.99,
        "MiMo 识别完成",
        f"✨ MiMo ({model}) 识别与对齐全部完成 ({len(todo)} 段)",
        tag="asr",
        label="ASR",
    )
    return {"language": detected, "segments": segments}

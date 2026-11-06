import os
import time
from typing import Any
import requests
import tempfile
from core.asr_backend.audio_preprocess import audio_slice_wav
from core.utils import load_key, report_progress, set_runtime_state as update_key

# ----------------------------------------
# ISO 639-2 to 1
# ----------------------------------------

iso_639_2_to_1 = {
    "eng": "en",
    "fra": "fr",
    "deu": "de",
    "ita": "it",
    "spa": "es",
    "rus": "ru",
    "kor": "ko",
    "jpn": "ja",
    "zho": "zh",
    "yue": "zh",
}

# ----------------------------
# elevenlabs format to whisper format
# ----------------------------

SPLIT_GAP = 1


def elev2whisper(elev_json, word_level_timestamp: bool = False) -> dict[str, Any]:
    words = elev_json.get("words", [])
    if not words:
        return {"segments": []}

    segments: list[dict[str, Any]] = []
    seg: dict[str, Any] = {
        "text": "",  # accumulated text
        "start": words[0]["start"],  # seg start time
        "end": words[0]["end"],  # seg end time (updates)
        "speaker_id": words[0]["speaker_id"],
        "words": [],  # optional per‑word info
    }

    for prev, nxt in zip(words, words[1:] + [None]):  # pairwise with sentinel
        seg["text"] += prev["text"]
        seg["end"] = prev["end"]
        if word_level_timestamp:
            seg["words"].append(
                {"word": prev["text"], "start": prev["start"], "end": prev["end"]}
            )
        # decide whether to break the segment
        if (
            nxt is None
            or (nxt["start"] - prev["end"] > SPLIT_GAP)
            or (nxt["speaker_id"] != seg["speaker_id"])
        ):
            seg["text"] = seg["text"].strip()
            if not word_level_timestamp:
                seg.pop("words")
            segments.append(seg)
            if nxt is not None:  # seed next segment
                seg = {
                    "text": "",
                    "start": nxt["start"],
                    "end": nxt["end"],
                    "speaker_id": nxt["speaker_id"],
                    "words": [],
                }
    return {"segments": segments}


def transcribe_audio_elevenlabs(raw_audio_path, vocal_audio_path, start=None, end=None):
    report_progress(
        0.10,
        "准备音频切片…",
        f"正在切分音频并准备上传至 ElevenLabs ({os.path.basename(str(vocal_audio_path))})",
        tag="asr",
        label="ASR",
    )

    # Use the same FFmpeg decoder as the other cloud ASR backend.
    audio_data = audio_slice_wav(vocal_audio_path, start, end)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filepath = temp_file.name
        temp_file.write(audio_data)

    try:
        api_key = load_key("whisper.elevenlabs_api_key")
        base_url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": api_key}

        whisper_lang = load_key("whisper.language")
        data = {
            "model_id": "scribe_v1",
            "timestamps_granularity": "word",
            "diarize": True,
            "num_speakers": None,
            "tag_audio_events": False,
        }
        if whisper_lang and whisper_lang != "auto":
            data["language_code"] = whisper_lang

        report_progress(
            0.35,
            "请求 ElevenLabs…",
            "正在请求 ElevenLabs Scribe 语音识别接口",
            tag="asr",
            label="ASR",
        )
        with open(temp_filepath, "rb") as audio_file:
            files = {"file": (os.path.basename(temp_filepath), audio_file, "audio/wav")}
            start_time = time.time()
            response = requests.post(base_url, headers=headers, data=data, files=files)

        response.raise_for_status()
        result = response.json()

        # save detected language
        raw_lang = result.get("language_code") or (
            whisper_lang if whisper_lang and whisper_lang != "auto" else "en"
        )
        detected_language = iso_639_2_to_1.get(raw_lang, raw_lang)
        update_key("whisper.detected_language", detected_language)

        # Adjust timestamps for all words by adding the start time
        if start is not None and "words" in result:
            for word in result["words"]:
                if "start" in word:
                    word["start"] += start
                if "end" in word:
                    word["end"] += start

        report_progress(
            0.98,
            "ElevenLabs 识别完成",
            f"ElevenLabs 语音识别完成 (耗时 {time.time() - start_time:.2f}s)",
            tag="asr",
            label="ASR",
        )
        # Keep word-level timestamps so downstream process_transcription has `words`.
        parsed_result = elev2whisper(result, word_level_timestamp=True)
        parsed_result["language"] = detected_language
        return parsed_result
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

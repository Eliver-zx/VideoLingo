"""Local Qwen3-ASR + Qwen3-ForcedAligner backend (default local ASR).

Dual track, like the WhisperX path: Qwen3-ASR transcribes the raw audio, then
Qwen3-ForcedAligner aligns that text against the vocal track (raw audio when
Demucs is off). The result uses WhisperX's segment/word structure so
process_transcription, the ASR cache and dubbing are unchanged.

Engines:
- mlx: Apple Silicon via mlx-audio (8-bit mlx-community weights).
- transformers: Windows / Linux / other Macs via the official qwen-asr package.
Heavy libraries are imported lazily so this module imports without them.
"""

from collections import Counter
import time

from core.asr_backend.qwen_asr.align_tokens import (
    _is_token_char as _is_token_char,
    attach_words as attach_words,
)
from core.asr_backend.qwen_asr.audio_windows import (
    load_audio_segment as load_audio_segment,
    match_length as match_length,
    split_windows as split_windows,
)
from core.asr_backend.qwen_asr.config import (
    CROSS_LANGUAGE_COVERAGE as CROSS_LANGUAGE_COVERAGE,
    CUT_SEARCH_SECONDS as CUT_SEARCH_SECONDS,
    EXPECTED_SCRIPTS as EXPECTED_SCRIPTS,
    ISO_TO_QWEN as ISO_TO_QWEN,
    JA_MIN_KANA_SHARE as JA_MIN_KANA_SHARE,
    MAX_NEW_TOKENS as MAX_NEW_TOKENS,
    MAX_WORD_LENGTH as MAX_WORD_LENGTH,
    MIN_WINDOW_SECONDS as MIN_WINDOW_SECONDS,
    MODELS as MODELS,
    PROBE_COVERAGE as PROBE_COVERAGE,
    PROBE_MIN_CHARS as PROBE_MIN_CHARS,
    PROBE_SECONDS as PROBE_SECONDS,
    PROBE_TOKEN_MARGIN as PROBE_TOKEN_MARGIN,
    PROBE_TOKENS_PER_SECOND as PROBE_TOKENS_PER_SECOND,
    PROBES_PER_WINDOW as PROBES_PER_WINDOW,
    QWEN_TO_ISO as QWEN_TO_ISO,
    REPEAT_COVERAGE as REPEAT_COVERAGE,
    REPEAT_MIN_CHARS as REPEAT_MIN_CHARS,
    REPROBE_SECONDS as REPROBE_SECONDS,
    REPROBES_PER_WINDOW as REPROBES_PER_WINDOW,
    RETRY_SEARCH_SECONDS as RETRY_SEARCH_SECONDS,
    RETRY_WINDOW_SECONDS as RETRY_WINDOW_SECONDS,
    SAMPLE_RATE as SAMPLE_RATE,
    SCRIPT_MIN_SHARE as SCRIPT_MIN_SHARE,
    SCRIPT_MIN_WEIGHT as SCRIPT_MIN_WEIGHT,
    SPARSE_CHARS_PER_SECOND as SPARSE_CHARS_PER_SECOND,
    TOKEN_MARGIN as TOKEN_MARGIN,
    TOKENS_PER_SECOND as TOKENS_PER_SECOND,
    WINDOW_SCRIPT_LIMITS as WINDOW_SCRIPT_LIMITS,
    WINDOW_SECONDS as WINDOW_SECONDS,
    ZH_MAX_KANA_SHARE as ZH_MAX_KANA_SHARE,
    _apple_silicon as _apple_silicon,
    _has_complete_weights as _has_complete_weights,
    _installed as _installed,
    _model_source as _model_source,
    iso_language as iso_language,
    model_size as model_size,
    primary_language as primary_language,
    qwen_language as qwen_language,
    resolve_engine as resolve_engine,
)
from core.asr_backend.qwen_asr.language_probe import (
    _vote as _vote,
    degeneration as degeneration,
    is_repetitive as is_repetitive,
    language_mismatch_error as language_mismatch_error,
    missed_speech as missed_speech,
    plan_languages as plan_languages,
    probe_mismatch as probe_mismatch,
    probe_ranges as probe_ranges,
    probe_token_budget as probe_token_budget,
    probe_window as probe_window,
    script_counts as script_counts,
    script_mismatch as script_mismatch,
    token_budget as token_budget,
    transcribe_window as transcribe_window,
)
from core.asr_backend.qwen_asr.runtime import (
    _align as _align,
    _free as _free,
    _torch_attn_implementation as _torch_attn_implementation,
    _torch_device as _torch_device,
    _torch_quantization_config as _torch_quantization_config,
    _transcribe as _transcribe,
    asr_session as asr_session,
)
from core.asr_backend.speech_edges import trim_speech_windows as trim_speech_windows
from core.utils import (
    check_cancel as check_cancel,
    except_handler as except_handler,
    load_key as load_key,
    load_key_or as load_key_or,
    log_event as log_event,
    report_progress as report_progress,
)


@except_handler("Qwen ASR processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    engine = resolve_engine()
    size = model_size()
    models = MODELS[engine]
    configured = load_key("whisper.language")
    forced = qwen_language(configured)
    report_progress(
        0.01,
        f"准备片段 {start:.0f}s–{end:.0f}s",
        f"准备 Qwen3-ASR {size} ({engine}) 识别区间 {start:.1f}s–{end:.1f}s",
        tag="asr",
        label="ASR",
    )

    raw = load_audio_segment(raw_audio_file, start, end)
    if vocal_audio_file == raw_audio_file:
        vocal = raw
    else:
        vocal = match_length(load_audio_segment(vocal_audio_file, start, end), len(raw))
    windows = [
        (a, b)
        for a, b in split_windows(raw)
        if b - a >= int(MIN_WINDOW_SECONDS * SAMPLE_RATE)
    ]
    windows = trim_speech_windows(raw, windows)

    t0 = time.time()
    pieces = []
    total_w = max(1, len(windows))
    report_progress(
        0.04,
        f"加载 Qwen3-ASR {size}",
        f"🚀 正在加载 Qwen3-ASR {size} ({engine})，共 {len(windows)} 个识别窗口...",
        tag="asr",
        label="ASR",
    )
    with asr_session(engine, models[size]) as run:
        plan = (
            [(forced, None)] * len(windows)
            if forced
            else plan_languages(run, raw, windows, progress_cb=report_progress)
        )
        for w_idx, ((a, b), (language, probes)) in enumerate(zip(windows, plan), 1):
            span_label = (
                f"{start + a / SAMPLE_RATE:.1f}s–{start + b / SAMPLE_RATE:.1f}s"
            )
            start_msg = f"🎙️ [{w_idx}/{total_w}] 正在识别窗口 {span_label} ({language or 'auto'})..."
            report_progress(
                0.15 + 0.58 * ((w_idx - 1) / total_w),
                f"语音识别 {w_idx}/{total_w} ({span_label})",
                start_msg,
                tag="asr",
                label="ASR",
            )
            t_win = time.time()
            window = transcribe_window(
                run, raw, a, b, language, probes, start, forced=bool(forced)
            )
            win_text = " ".join(text for *_, text in window).strip()
            preview = (
                (win_text[:48] + "…") if len(win_text) > 48 else (win_text or "<静音>")
            )
            done_msg = (
                f"✅ [{w_idx}/{total_w}] 完成窗口 {span_label} "
                f"({time.time() - t_win:.1f}s, {len(win_text)} 字符): {preview}"
            )
            report_progress(
                0.15 + 0.58 * (w_idx / total_w),
                f"已完成窗口 {w_idx}/{total_w}",
                done_msg,
                tag="asr",
                label="ASR",
            )
            if forced:
                mismatch = script_mismatch(win_text, configured, **WINDOW_SCRIPT_LIMITS)
                if mismatch:
                    raise language_mismatch_error(
                        f"{start + a / SAMPLE_RATE:.1f}-{start + b / SAMPLE_RATE:.1f}s",
                        forced,
                        mismatch,
                    )
            pieces.extend(window)
    log_event(f"Qwen3-ASR 文本识别耗时 {time.time() - t0:.2f}s", tag="asr", label="ASR")

    if forced:
        mismatch = script_mismatch(" ".join(text for *_, text in pieces), configured)
        if mismatch:
            raise language_mismatch_error(f"{start:.1f}-{end:.1f}s", forced, mismatch)
        detected = configured
    else:
        votes: Counter[str] = Counter()
        for a, b, lang, text in pieces:
            iso = iso_language(lang)
            if text and iso:
                votes[iso] += b - a
        detected = max(votes, key=lambda k: votes[k]) if votes else None
        if not detected and any(text for *_, text in pieces):
            raise ValueError(
                "Qwen3-ASR could not detect the language; set the recognition language and retry."
            )

    t0 = time.time()
    default_qwen_lang = ISO_TO_QWEN.get(detected or "") or "English"
    todo = [
        (a, b, text, (primary_language(lang) or default_qwen_lang))
        for a, b, lang, text in pieces
        if text
    ]
    jobs = [(vocal[a:b], text, lang) for a, b, text, lang in todo]
    aligned = _align(engine, models["aligner"], jobs) if jobs else []
    log_event(
        f"ForcedAligner 词级对齐耗时 {time.time() - t0:.2f}s", tag="asr", label="ASR"
    )
    report_progress(
        0.99,
        "语音识别完成",
        f"✨ Qwen3-ASR 识别与对齐全部完成 ({len(todo)} 段)",
        tag="asr",
        label="ASR",
    )

    segments = []
    for (a, b, text, _), items in zip(todo, aligned):
        words = attach_words(
            text, items, start + a / SAMPLE_RATE, limit=(b - a) / SAMPLE_RATE
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
    return {"language": detected, "segments": segments}

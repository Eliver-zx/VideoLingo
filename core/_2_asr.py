from core.utils import (
    check_cancel,
    check_file_exists,
    load_key,
    load_key_or,
    log_event,
    progress_scope,
    report_progress,
    set_runtime_state as update_key,
)
from core.asr_backend.audio_preprocess import (
    process_transcription,
    convert_video_to_audio,
    prepare_audio_for_asr,
    split_audio,
    save_results,
)
from core._1_ytdlp import find_media_file
from core.utils.models import (
    _RAW_AUDIO_FILE,
    _VOCAL_AUDIO_FILE,
    resolve_task_path,
)
from core.asr_backend import transcription_cache as cache
from core.workspace import get_active_workspace


def _needs_high_quality_audio(demucs: bool) -> bool:
    """Return True when vocal separation or dubbing requires stereo high-sample-rate audio."""
    if demucs:
        return True
    ws = get_active_workspace()
    if ws.manifest_path.is_file():
        exec_cfg = ws.load_manifest().get("execution") or {}
        if (
            exec_cfg.get("dubbing_enabled")
            or exec_cfg.get("desired_stage") == "dubbing"
        ):
            return True
    return False


def local_backend(whisper):
    """Local ASR backend: "qwen" (default) or "whisperx" (install the packages yourself)."""
    backend = str(whisper.get("backend", "qwen")).lower()
    if backend not in ("qwen", "whisperx"):
        raise ValueError(
            f"whisper.backend must be 'qwen' or 'whisperx', got {backend!r}"
        )
    return backend


WHISPERX_NOT_INSTALLED = (
    "WhisperX is not installed. "
    "Install the whisperx packages yourself, or set whisper.backend to qwen."
)


def prepare_audio(media_file, media_type, demucs):
    """The audio of the input for the recognition and the dubbing. Returns the audio with the voice."""
    high_quality = _needs_high_quality_audio(bool(demucs))
    with progress_scope(
        0.0, 0.30 if demucs else 0.98, "提取音频", tag="asr", label="FFMPEG"
    ):
        if media_type == "video":
            convert_video_to_audio(media_file, high_quality=high_quality)
        else:
            prepare_audio_for_asr(media_file, high_quality=high_quality)

    if not demucs:
        return resolve_task_path(_RAW_AUDIO_FILE)
    from importlib.util import find_spec
    from core.asr_backend.demucs_vl import demucs_audio, separation_backend_package

    pkg = separation_backend_package()
    if find_spec(pkg) is None:
        raise RuntimeError(
            f"Vocal separation backend '{pkg}' is not installed. "
            "Turn vocal separation off in settings or install the required package."
        )
    with progress_scope(0.30, 0.98, "人声分离", tag="asr", label="DEMUCS"):
        demucs_audio()
    return resolve_task_path(_VOCAL_AUDIO_FILE)


SUPPORTED_RUNTIMES = (
    "local",
    "mimo",
    "elevenlabs",
    "mai",
    "deepgram",
    "assembly",
    "zero",
    "replicate_parakeet",
    "replicate_whisper",
)


@check_file_exists("asr")
def transcribe():
    raw_runtime = str(load_key("whisper.runtime") or "").strip()
    runtime = "zero" if raw_runtime == "Zero" else raw_runtime
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(
            "Select local, mimo, elevenlabs, mai, deepgram, assembly, zero, replicate_parakeet or replicate_whisper "
            "for whisper.runtime. The 302.ai WhisperX cloud service has been retired."
        )
    if runtime == "local" and local_backend(load_key("whisper")) == "whisperx":
        from importlib.util import find_spec

        if find_spec("whisperx") is None:
            raise RuntimeError(WHISPERX_NOT_INSTALLED)
    if runtime == "mimo":
        from core.asr_backend.mimo_asr import configured_credentials

        configured_credentials()
    elif runtime == "mai":
        from core.asr_backend.mai_asr import configured_credentials

        configured_credentials()
    elif runtime == "deepgram":
        from core.asr_backend.deepgram_asr import configured_credentials

        configured_credentials()
    elif runtime == "assembly":
        from core.asr_backend.assembly_asr import configured_credentials

        configured_credentials()
    elif runtime == "zero":
        from core.asr_backend.zero_asr import configured_credentials

        configured_credentials()
    elif runtime in ("replicate_parakeet", "replicate_whisper"):
        from core.asr_backend.replicate_asr import configured_credentials

        configured_credentials()
    # 1. prepare audio
    media_file, media_type = find_media_file()
    whisper = dict(load_key("whisper"))
    whisper["runtime"] = runtime
    demucs = load_key("demucs")
    if demucs:
        from core.asr_backend.demucs_vl import resolve_separation_model

        whisper["vocal_separation_model"] = resolve_separation_model(
            load_key_or("vocal_separation_model", "scnet")
        )
    if runtime == "local":
        whisper["backend"] = local_backend(whisper)
        if whisper["backend"] == "qwen":
            from core.asr_backend import qwen_asr_local

            whisper["qwen_model"] = qwen_asr_local.model_size(whisper.get("qwen_model"))
            whisper["qwen_engine"] = qwen_asr_local.resolve_engine(
                whisper.get("qwen_engine")
            )
    elif runtime == "mimo":
        from core.asr_backend import qwen_asr_local
        from core.asr_backend.mimo_asr import DEFAULT_MIMO_MODEL

        whisper["mimo_model"] = str(
            whisper.get("mimo_model") or DEFAULT_MIMO_MODEL
        ).strip()
        whisper["qwen_engine"] = qwen_asr_local.resolve_engine(
            whisper.get("qwen_engine")
        )
    report_progress(
        0.01,
        "校验媒体指纹…",
        "正在计算媒体文件指纹并检查转录缓存",
        tag="asr",
        label="CACHE",
    )
    key = (
        cache.cache_key(media_file, whisper, demucs)
        if whisper.get("cache", True)
        else None
    )
    cached = cache.read_result(key, "complete") if key else None
    with progress_scope(0.03, 0.24, "音频预处理", tag="asr", label="FFMPEG"):
        vocal_audio = prepare_audio(media_file, media_type, demucs)

    # Downstream alignment/dubbing still needs the prepared audio on a cache hit.
    if cached:
        check_cancel()
        update_key("whisper.detected_language", cached["language"])
        save_results(process_transcription(cached["result"]))
        report_progress(
            1.0,
            "命中转录缓存",
            "命中本地转录缓存，已复用识别结果",
            tag="asr",
            label="CACHE",
        )
        return

    # 3. Extract audio
    raw_audio_file = resolve_task_path(_RAW_AUDIO_FILE)
    with progress_scope(0.24, 0.30, "静音切分", tag="asr", label="VAD"):
        if (
            runtime == "mai"
            and str(whisper.get("mai_provider", "azure")).lower() == "openrouter"
        ):
            # OpenRouter's upstream has a roughly 60-second processing timeout.
            segments = split_audio(raw_audio_file, target_len=120, win=15)
        else:
            segments = split_audio(raw_audio_file)

    # 4. Transcribe audio by clips
    all_results = []
    language = None
    if runtime == "local" and whisper["backend"] == "qwen":
        from core.asr_backend.qwen_asr_local import transcribe_audio as ts

        log_event(
            f"启动本地 Qwen3-ASR ({whisper['qwen_model']}) + ForcedAligner 语音识别",
            tag="asr",
            label="ASR",
        )
    elif runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts

        log_event("启动本地 WhisperX 语音识别", tag="asr", label="ASR")
    elif runtime == "mimo":
        from core.asr_backend.mimo_asr import transcribe_audio_mimo as ts

        log_event(
            f"启动小米 MiMo ({whisper['mimo_model']}) + ForcedAligner 语音识别",
            tag="asr",
            label="ASR",
        )
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts

        log_event("启动 ElevenLabs Scribe 语音识别", tag="asr", label="ASR")
    elif runtime == "mai":
        from core.asr_backend.mai_asr import (
            transcribe_audio_mai as ts,
            selected_provider,
        )

        log_event(
            f"启动 MAI-Transcribe-2 ({selected_provider()}) 语音识别",
            tag="asr",
            label="ASR",
        )
    elif runtime == "deepgram":
        from core.asr_backend.deepgram_asr import transcribe_audio_deepgram as ts

        log_event("启动 Deepgram Nova 语音识别", tag="asr", label="ASR")
    elif runtime == "assembly":
        from core.asr_backend.assembly_asr import transcribe_audio_assembly as ts

        log_event("启动 AssemblyAI 语音识别", tag="asr", label="ASR")
    elif runtime == "zero":
        from core.asr_backend.zero_asr import transcribe_audio_zero as ts

        log_event("启动 Gladia Zero 语音识别", tag="asr", label="ASR")
    elif runtime == "replicate_parakeet":
        from core.asr_backend.replicate_asr import (
            transcribe_audio_replicate_parakeet as ts,
        )

        log_event(
            "启动 Replicate Parakeet-TDT-0.6b-v2 语音识别", tag="asr", label="ASR"
        )
    elif runtime == "replicate_whisper":
        from core.asr_backend.replicate_asr import (
            transcribe_audio_replicate_whisper as ts,
        )

        log_event("启动 Replicate WhisperX 语音识别", tag="asr", label="ASR")
    else:
        raise ValueError(f"Unsupported ASR runtime: {runtime}")

    total_seg = max(1, len(segments))
    for idx, (start, end) in enumerate(segments, 1):
        check_cancel()
        seg_lo = 0.30 + 0.66 * ((idx - 1) / total_seg)
        seg_hi = 0.30 + 0.66 * (idx / total_seg)
        seg_prefix = f"片段 {idx}/{total_seg}" if total_seg > 1 else ""
        part = f"{start}_{end}"
        cached = cache.read_result(key, part) if key else None
        if cached:
            result = cached["result"]
            language = cached["language"]
            report_progress(
                seg_hi,
                f"{seg_prefix} 命中缓存".strip(),
                f"音频片段 [{idx}/{total_seg}] ({start:.1f}s-{end:.1f}s) 命中转录缓存",
                tag="asr",
                label="CACHE",
            )
        else:
            with progress_scope(seg_lo, seg_hi, seg_prefix, tag="asr", label="ASR"):
                result = ts(raw_audio_file, vocal_audio, start, end)
            check_cancel()
            language = (
                whisper["language"]
                if whisper["language"] != "auto"
                else result.get("language")
            )
            if key:
                cache.write_result(key, part, result, language)
        if language:
            update_key("whisper.detected_language", language)
        all_results.append(result)

    # 5. Combine results
    report_progress(
        0.97, "合并识别结果…", "正在合并词级时间戳并写入工作区", tag="asr", label="ASR"
    )
    combined_result = {"segments": []}
    for result in all_results:
        combined_result["segments"].extend(result["segments"])

    # 6. Process df
    df = process_transcription(combined_result)
    check_cancel()
    if df.empty:
        # Silence (or a wrong recognition language) yields no words; later steps need at least one.
        raise ValueError(
            "No speech was recognized in the audio. Check the source audio, "
            "or set the recognition language explicitly and retry."
        )
    save_results(df)
    if key:
        cache.write_result(key, "complete", combined_result, language)
    report_progress(
        1.0,
        f"共识别 {len(df)} 词",
        f"语音识别完成，共保存 {len(df)} 个词级时间戳",
        tag="asr",
        label="ASR",
    )


if __name__ == "__main__":
    transcribe()

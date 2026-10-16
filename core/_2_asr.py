from core.utils import *
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, prepare_audio_for_asr, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_media_file
from core.utils.models import *
from core.asr_backend import transcription_cache as cache

def local_backend(whisper):
    """Local ASR backend: "qwen" (default) or "whisperx" (install the packages yourself)."""
    backend = str(whisper.get("backend", "qwen")).lower()
    if backend not in ("qwen", "whisperx"):
        raise ValueError(f"whisper.backend must be 'qwen' or 'whisperx', got {backend!r}")
    return backend

WHISPERX_NOT_INSTALLED = (
    "WhisperX is not installed. "
    "Follow the manual page (docs/pages/docs/whisperx-manual.en-US.md) "
    "and install the extra packages yourself, or set whisper.backend to qwen."
)

def prepare_audio(media_file, media_type, demucs):
    """The audio of the input for the recognition and the dubbing. Returns the audio with the voice."""
    if media_type == "video":
        convert_video_to_audio(media_file)
    else:
        prepare_audio_for_asr(media_file)

    # Vocal separation:
    if not demucs:
        return _RAW_AUDIO_FILE
    from importlib.util import find_spec
    from core.asr_backend.demucs_vl import demucs_audio, separation_backend_package
    pkg = separation_backend_package()
    if find_spec(pkg) is None:
        raise RuntimeError(
            f"Vocal separation backend '{pkg}' is not installed. "
            "Turn vocal separation off in settings or install the required package."
        )
    demucs_audio()
    return normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")

SUPPORTED_RUNTIMES = (
    "local", "elevenlabs", "mai",
    "deepgram", "assembly", "zero", "replicate_parakeet", "replicate_whisper",
)

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    raw_runtime = str(load_key("whisper.runtime") or "").strip()
    runtime = "zero" if raw_runtime == "Zero" else raw_runtime
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(
            "Select local, elevenlabs, mai, deepgram, assembly, zero, replicate_parakeet or replicate_whisper "
            "for whisper.runtime. The 302.ai WhisperX cloud service has been retired."
        )
    if runtime == "local" and local_backend(load_key("whisper")) == "whisperx":
        from importlib.util import find_spec
        if find_spec("whisperx") is None:
            raise RuntimeError(WHISPERX_NOT_INSTALLED)
    if runtime == "mai":
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
            load_key_or("vocal_separation_model", "htdemucs")
        )
    if runtime == "local":
        whisper["backend"] = local_backend(whisper)
        if whisper["backend"] == "qwen":
            from core.asr_backend import qwen_asr_local
            whisper["qwen_model"] = qwen_asr_local.model_size(whisper.get("qwen_model"))
            whisper["qwen_engine"] = qwen_asr_local.resolve_engine(whisper.get("qwen_engine"))
    key = cache.cache_key(media_file, whisper, demucs) if whisper.get("cache", True) else None
    cached = cache.read_result(key, "complete") if key else None
    vocal_audio = prepare_audio(media_file, media_type, demucs)

    # Downstream alignment/dubbing still needs the prepared audio on a cache hit.
    if cached:
        check_cancel()
        update_key("whisper.detected_language", cached["language"])
        save_results(process_transcription(cached["result"]))
        rprint("[green]Reused transcription from the content cache.[/green]")
        return

    # 3. Extract audio
    if runtime == "mai" and str(whisper.get("mai_provider", "azure")).lower() == "openrouter":
        # OpenRouter's upstream has a roughly 60-second processing timeout.
        segments = split_audio(_RAW_AUDIO_FILE, target_len=120, win=15)
    else:
        segments = split_audio(_RAW_AUDIO_FILE)
    
    # 4. Transcribe audio by clips
    all_results = []
    language = None
    if runtime == "local" and whisper["backend"] == "qwen":
        from core.asr_backend.qwen_asr_local import transcribe_audio as ts
        rprint(f"[cyan]🎤 Transcribing audio with local Qwen3-ASR {whisper['qwen_model']} + ForcedAligner...[/cyan]")
    elif runtime == "local":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with local WhisperX...[/cyan]")
    elif runtime == "elevenlabs":
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as ts
        rprint("[cyan]🎤 Transcribing audio with ElevenLabs API...[/cyan]")
    elif runtime == "mai":
        from core.asr_backend.mai_asr import transcribe_audio_mai as ts, selected_provider
        rprint(f"[cyan]🎤 Transcribing audio with MAI-Transcribe-2 via {selected_provider()}...[/cyan]")
    elif runtime == "deepgram":
        from core.asr_backend.deepgram_asr import transcribe_audio_deepgram as ts
        rprint("[cyan]🎤 Transcribing audio with Deepgram API...[/cyan]")
    elif runtime == "assembly":
        from core.asr_backend.assembly_asr import transcribe_audio_assembly as ts
        rprint("[cyan]🎤 Transcribing audio with AssemblyAI API...[/cyan]")
    elif runtime == "zero":
        from core.asr_backend.zero_asr import transcribe_audio_zero as ts
        rprint("[cyan]🎤 Transcribing audio with Gladia Zero API...[/cyan]")
    elif runtime == "replicate_parakeet":
        from core.asr_backend.replicate_asr import transcribe_audio_replicate_parakeet as ts
        rprint("[cyan]🎤 Transcribing audio with Replicate Parakeet-TDT-0.6b-v2...[/cyan]")
    elif runtime == "replicate_whisper":
        from core.asr_backend.replicate_asr import transcribe_audio_replicate_whisper as ts
        rprint("[cyan]🎤 Transcribing audio with Replicate WhisperX...[/cyan]")
    else:
        raise ValueError(f"Unsupported ASR runtime: {runtime}")

    for start, end in segments:
        check_cancel()
        part = f"{start}_{end}"
        cached = cache.read_result(key, part) if key else None
        if cached:
            result = cached["result"]
            language = cached["language"]
        else:
            result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
            check_cancel()
            language = whisper["language"] if whisper["language"] != "auto" else result.get("language")
            if key:
                cache.write_result(key, part, result, language)
        if language:
            update_key("whisper.detected_language", language)
        all_results.append(result)
    
    # 5. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. Process df
    df = process_transcription(combined_result)
    check_cancel()
    if df.empty:
        # Silence (or a wrong recognition language) yields no words; later steps need at least one.
        raise ValueError("No speech was recognized in the audio. Check the source audio, "
                         "or set the recognition language explicitly and retry.")
    save_results(df)
    if key:
        cache.write_result(key, "complete", combined_result, language)
        
if __name__ == "__main__":
    transcribe()

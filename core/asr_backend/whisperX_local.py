import functools
from pathlib import Path
import time
import warnings

from core.utils import (
    except_handler,
    load_key,
    log_event,
    report_progress,
    set_runtime_state as update_key,
)

warnings.filterwarnings("ignore")

_whisperx_initialized = False


def _ensure_whisperx():
    global _whisperx_initialized
    import torch  # type: ignore[import-not-found]

    if not _whisperx_initialized:
        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_torch_load(*args, **kwargs):
            if kwargs.get("weights_only") is None:
                kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
        _whisperx_initialized = True

    import whisperx  # type: ignore[import-not-found]
    from whisperx.audio import (  # type: ignore[import-not-found]
        SAMPLE_RATE as _WHISPERX_SR,
        load_audio as _whisperx_load_audio,
    )

    return torch, whisperx, _whisperx_load_audio, _WHISPERX_SR


def _complete_model_directory(path):
    required = ("config.json", "model.bin", "tokenizer.json")
    return all(
        (Path(path) / name).is_file() and (Path(path) / name).stat().st_size > 0
        for name in required
    )


def resolve_whisper_model(model_name, model_dir=None):
    """Resolve complete local files before allowing any Hub request."""
    from faster_whisper.utils import download_model  # type: ignore[import-not-found]
    from huggingface_hub.errors import LocalEntryNotFoundError

    if model_dir is None:
        model_dir = load_key("model_dir")
    explicit = Path(model_name)
    project = Path(model_dir) / model_name
    for directory in (explicit, project):
        if directory.is_dir():
            if not _complete_model_directory(directory):
                raise ValueError(
                    f"Local model directory is incomplete: {directory}. "
                    "Expected non-empty config.json, model.bin and tokenizer.json."
                )
            log_event(
                f"使用本地 Whisper 模型: {directory.resolve()}",
                tag="asr",
                label="ASR",
            )
            return str(directory.resolve())

    # The installed loader owns aliases (large/turbo/distil) and HF cache locations.
    # local_files_only resolves the cached revision without checking the network.
    for cache_dir in (str(model_dir), None):
        try:
            snapshot = download_model(
                model_name, cache_dir=cache_dir, local_files_only=True
            )
        except LocalEntryNotFoundError:
            continue
        if _complete_model_directory(snapshot):
            log_event(f"复用缓存 Whisper 模型: {snapshot}", tag="asr", label="ASR")
            return snapshot

    report_progress(
        0.02,
        "下载 Whisper 模型…",
        f"本地未找到完整 Whisper 模型，正在从 HuggingFace 下载 {model_name}",
        tag="asr",
        label="ASR",
    )
    snapshot = download_model(model_name)
    if not _complete_model_directory(snapshot):
        raise RuntimeError("Downloaded Whisper model is incomplete")
    log_event(f"Whisper 模型准备就绪: {snapshot}", tag="asr", label="ASR")
    return snapshot


@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    torch, whisperx, _whisperx_load_audio, _WHISPERX_SR = _ensure_whisperx()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report_progress(
        0.05,
        f"初始化 WhisperX ({device})…",
        f"启动 WhisperX 语音识别 (设备: {device}, 区间 {start:.1f}s–{end:.1f}s)",
        tag="asr",
        label="ASR",
    )

    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        log_event(
            f"GPU 显存: {gpu_mem:.2f} GB, Batch size: {batch_size}, 精度: {compute_type}",
            tag="asr",
            label="ASR",
        )
    else:
        batch_size = 1
        compute_type = "int8"
        log_event(
            f"CPU 模式 Batch size: {batch_size}, 精度: {compute_type}",
            tag="asr",
            label="ASR",
        )

    model_dir = load_key("model_dir")
    if WHISPER_LANGUAGE == "zh":
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = Path(model_dir) / "Belle-whisper-large-v3-zh-punct-fasterwhisper"
        if local_model.is_dir():
            model_name = str(local_model.resolve())
    else:
        model_name = load_key("whisper.model")
        if "turbo" in str(model_name).lower():
            # turbo loops on repeated words in this pipeline, so it is rewritten.
            log_event(
                "large-v3-turbo 易产生重复循环，自动切换为 large-v3",
                tag="warn",
                label="ASR",
                level="warn",
            )
            model_name = "large-v3"
    model_name = resolve_whisper_model(model_name, model_dir)

    vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}
    asr_options = {
        "temperatures": [0],
        "initial_prompt": "",
    }
    whisper_language = None if "auto" in WHISPER_LANGUAGE else WHISPER_LANGUAGE
    load_kwargs = dict(
        device=device,
        compute_type=compute_type,
        language=whisper_language,
        vad_options=vad_options,
        asr_options=asr_options,
        local_files_only=True,
    )
    model = whisperx.load_model(model_name, **load_kwargs)

    def load_audio_segment(audio_file, start, end):
        # Use whisperx's ffmpeg-based loader instead of librosa.load() which
        # deadlocks inside Streamlit's ScriptRunner thread.
        full_audio = _whisperx_load_audio(audio_file, sr=_WHISPERX_SR)
        start_sample = int(start * _WHISPERX_SR)
        end_sample = int(end * _WHISPERX_SR)
        return full_audio[start_sample:end_sample]

    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)

    # -------------------------
    # 1. transcribe raw audio
    # -------------------------
    transcribe_start_time = time.time()
    report_progress(
        0.20,
        f"WhisperX 识别 ({start:.0f}s–{end:.0f}s)…",
        f"正在运行 WhisperX 语音识别 ({start:.1f}s–{end:.1f}s)",
        tag="asr",
        label="ASR",
    )
    result = model.transcribe(
        raw_audio_segment, batch_size=batch_size, print_progress=False
    )
    transcribe_time = time.time() - transcribe_start_time
    report_progress(
        0.70,
        "WhisperX 识别完成",
        f"WhisperX 文本识别完成 (耗时 {transcribe_time:.2f}s)",
        tag="asr",
        label="ASR",
    )

    # Free GPU resources
    del model
    torch.cuda.empty_cache()

    # Save language
    detected_language = result["language"]
    update_key("whisper.detected_language", detected_language)
    if result["language"] == "zh" and WHISPER_LANGUAGE != "zh":
        raise ValueError(
            "Please specify the transcription language as zh and try again!"
        )

    # -------------------------
    # 2. align by vocal audio
    # -------------------------
    align_start_time = time.time()
    report_progress(
        0.75,
        "词级时间戳对齐…",
        f"正在加载 WhisperX 对齐模型并对齐词级时间戳 ({detected_language})",
        tag="asr",
        label="ASR",
    )
    # Align timestamps using vocal audio
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        vocal_audio_segment,
        device,
        return_char_alignments=False,
    )
    result["language"] = detected_language
    align_time = time.time() - align_start_time
    report_progress(
        0.99,
        "WhisperX 对齐完成",
        f"WhisperX 词级时间戳对齐完成 (耗时 {align_time:.2f}s)",
        tag="asr",
        label="ASR",
    )

    # Free GPU resources again
    torch.cuda.empty_cache()
    del model_a

    # Adjust timestamps
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start
        for word in segment["words"]:
            if "start" in word:
                word["start"] += start
            if "end" in word:
                word["end"] += start
    return result

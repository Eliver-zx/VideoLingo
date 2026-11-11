"""Configuration, model registry, weight cache validation, and language tables for Qwen3-ASR."""

import importlib.util
import json
import os
from pathlib import Path
import platform

SAMPLE_RATE = 16000
# Qwen3-ForcedAligner accepts at most 180 s per call (qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS).
# ASR uses the same windows so each transcript aligns against exactly its own audio.
WINDOW_SECONDS = 180
# Cut each window at the quietest 100 ms inside its last 30 s.
CUT_SEARCH_SECONDS = 30
MAX_NEW_TOKENS = 2400
# Token budget by clip length, so a looping clip stops early instead of generating a fixed
# maximum. Transcripts: the fastest Chinese/Japanese speech (~8 chars/s, most common
# characters one token, plus punctuation) needs ~6-8 tokens/s, English far less; 13/s is
# 1.6x that, and a 180 s window gets 2372 tokens (the old flat cap was 2048).
TOKENS_PER_SECOND = 13
TOKEN_MARGIN = 32
# Probes only decide the language and give a size reference, so they get a tighter budget:
# 20 s -> 116 tokens (~100-150 CJK chars or ~90 English words).
PROBE_TOKENS_PER_SECOND = 5
PROBE_TOKEN_MARGIN = 16
# Shorter windows are dropped; MLX would otherwise zero-pad anything under its 1 s default.
MIN_WINDOW_SECONDS = 0.1

# Language detection with whisper.language = auto: probe short clips spread over the window,
# vote, then transcribe the window with that language forced.
PROBE_SECONDS = 20
PROBES_PER_WINDOW = 3

# Degenerate output thresholds and retry window sizes.
REPEAT_MIN_CHARS = 40
REPEAT_COVERAGE = 0.5
PROBE_MIN_CHARS = 30
PROBE_COVERAGE = 0.5
RETRY_WINDOW_SECONDS = 60
RETRY_SEARCH_SECONDS = 15
SPARSE_CHARS_PER_SECOND = 2.0
CROSS_LANGUAGE_COVERAGE = 0.25
REPROBE_SECONDS = 10
REPROBES_PER_WINDOW = 6
MAX_WORD_LENGTH = 30  # process_transcription drops longer words

# Scripts a transcript in each language is written in. Other languages Qwen3-ASR knows use Latin.
EXPECTED_SCRIPTS: dict[str, set[str]] = {
    "zh": {"Han"},
    "yue": {"Han"},
    "ja": {"Han", "Kana"},
    "ko": {"Hangul"},
    "ru": {"Cyrillic"},
    "mk": {"Cyrillic"},
    "el": {"Greek"},
    "ar": {"Arabic"},
    "fa": {"Arabic"},
    "hi": {"Devanagari"},
    "th": {"Thai"},
}
_SCRIPT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("CJK", "Han"),
    ("IDEOGRAPHIC", "Han"),
    ("HIRAGANA", "Kana"),
    ("KATAKANA", "Kana"),
    ("HALFWIDTH KATAKANA", "Kana"),
    ("HANGUL", "Hangul"),
    ("HALFWIDTH HANGUL", "Hangul"),
    ("LATIN", "Latin"),
    ("FULLWIDTH LATIN", "Latin"),
    ("CYRILLIC", "Cyrillic"),
    ("GREEK", "Greek"),
    ("ARABIC", "Arabic"),
    ("DEVANAGARI", "Devanagari"),
    ("THAI", "Thai"),
)
_SCRIPT_WEIGHT = {"Han": 3, "Kana": 3, "Hangul": 3}
SCRIPT_MIN_WEIGHT = 100
SCRIPT_MIN_SHARE = 0.2
JA_MIN_KANA_SHARE = 0.05
ZH_MAX_KANA_SHARE = 0.2
WINDOW_SCRIPT_LIMITS = {
    "min_weight": 300,
    "min_share": 0.05,
    "ja_min_kana": 0.02,
    "zh_max_kana": 0.5,
}

MODELS = {
    "mlx": {
        "1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "aligner": "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
    },
    "transformers": {
        "1.7b": "Qwen/Qwen3-ASR-1.7B",
        "0.6b": "Qwen/Qwen3-ASR-0.6B",
        "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    },
}

# ISO 639-1 (whisper.language / detected_language) <-> Qwen3-ASR language names.
ISO_TO_QWEN = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}
QWEN_TO_ISO = {name: code for code, name in ISO_TO_QWEN.items()}
# Downstream splitting/joining only knows "zh"; Cantonese is written in Chinese characters.
QWEN_TO_ISO["Cantonese"] = "zh"


def _facade():
    from core.asr_backend import qwen_asr_local

    return qwen_asr_local


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def model_size(requested: str | None = None) -> str:
    f = _facade()
    size = str(requested or f.load_key_or("whisper.qwen_model", "1.7b")).lower()
    if size not in ("1.7b", "0.6b"):
        raise ValueError(f"whisper.qwen_model must be '1.7b' or '0.6b', got {size!r}")
    return size


def resolve_engine(requested: str | None = None) -> str:
    """auto -> MLX on Apple Silicon (qwen-asr only if it is what this environment has), else transformers."""
    f = _facade()
    engine = str(requested or f.load_key_or("whisper.qwen_engine", "auto")).lower()
    if engine not in ("auto", "mlx", "transformers"):
        raise ValueError(
            f"whisper.qwen_engine must be 'auto', 'mlx' or 'transformers', got {engine!r}"
        )
    if engine == "auto":
        # Apple Silicon requirements install mlx-audio, not qwen-asr; point a broken install back to it.
        mlx = f._apple_silicon() and (
            f._installed("mlx_audio") or not f._installed("qwen_asr")
        )
        engine = "mlx" if mlx else "transformers"
    module = "mlx_audio" if engine == "mlx" else "qwen_asr"
    if not f._installed(module):
        package = "mlx-audio" if engine == "mlx" else "qwen-asr"
        raise ImportError(
            f"Qwen ASR engine '{engine}' needs the '{package}' package. "
            "Run Start.command / Start.bat or `uv pip install -r requirements.txt` to install it."
        )
    return engine


def _has_complete_weights(local: Path) -> bool:
    try:
        if not local.is_dir():
            return False
        resolved = local.resolve()
        with os.scandir(resolved) as it:
            file_names = {entry.name for entry in it if entry.is_file()}
    except OSError:
        return False

    if "config.json" not in file_names:
        return False

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if index_name in file_names:
            if (
                "tokenizer_config.json" not in file_names
                or "preprocessor_config.json" not in file_names
            ):
                return False
            try:
                data = json.loads((resolved / index_name).read_text(encoding="utf-8"))
                shards = set((data.get("weight_map") or {}).values())
            except (OSError, ValueError, TypeError, AttributeError):
                return False
            if not shards or any(not (resolved / shard).is_file() for shard in shards):
                return False
            return not any(resolved.rglob("*.incomplete"))

    if any(
        not name.startswith(".")
        and "-of-" in name
        and name.endswith((".safetensors", ".bin"))
        for name in file_names
    ):
        return False
    if not any(
        not name.startswith(".") and name.endswith((".safetensors", ".bin", ".npz"))
        for name in file_names
    ):
        return False
    return not any(resolved.rglob("*.incomplete"))


def _model_source(repo_id: str) -> str:
    """Ensure a complete local copy in model_dir (e.g. _model_cache/Qwen3-ASR-1.7B), downloading into it if missing."""
    f = _facade()
    raw_dir = str(f.load_key("model_dir") or "./_model_cache").strip().strip("\"'")
    short_name = repo_id.split("/")[-1]
    local = Path(raw_dir) / short_name
    if f._has_complete_weights(local):
        f.log_event(
            f"使用本地缓存模型: {local.name}",
            tag="asr",
            label="ASR",
        )
        return str(local.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return repo_id
    local.mkdir(parents=True, exist_ok=True)
    msg = f"⬇️ Downloading {repo_id} into {local.resolve()} ..."
    f.report_progress(
        sub_label=f"下载模型 {short_name}…", message=msg, tag="asr", label="ASR"
    )
    snapshot_download(repo_id=repo_id, local_dir=str(local.resolve()))
    return str(local.resolve())


def qwen_language(language: str | None) -> str | None:
    """Map whisper.language to a Qwen language name; None lets Qwen detect it."""
    if not language or language == "auto":
        return None
    if language not in ISO_TO_QWEN:
        raise ValueError(
            f"Qwen3-ASR does not support language '{language}'. Supported: {sorted(ISO_TO_QWEN)}"
        )
    return ISO_TO_QWEN[language]


def primary_language(name: str | None) -> str | None:
    """Qwen may report 'Chinese,English' for code-switched audio; the first language is primary."""
    first = (name or "").split(",")[0].strip()
    first = first[:1].upper() + first[1:].lower()
    return first if first in QWEN_TO_ISO else None


def iso_language(name: str | None) -> str | None:
    prim = primary_language(name)
    return QWEN_TO_ISO.get(prim) if prim else None

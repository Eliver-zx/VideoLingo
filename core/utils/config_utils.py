from __future__ import annotations

import contextlib
import contextvars
import copy
import os
from pathlib import Path
import threading
from typing import Any
import unicodedata
from ruamel.yaml import YAML

CONFIG_PATH = 'config.yaml'
lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

_TASK_CONFIG_OVERRIDES: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_TASK_CONFIG_OVERRIDES", default=None
)
_PROCESS_CONFIG_OVERRIDES: dict[str, Any] | None = None
_OVERRIDES_LOCK = threading.Lock()

PLACEHOLDER_KEY_PREFIXES = ("your_", "YOUR_")

DEFAULT_CONFIG: dict[str, Any] = {
    "display_language": "zh-CN",
    "api": {
        "key": "",
        "base_url": "https://api.openlux.ai/v1",
        "model": "gpt-6-luna",
        "llm_support_json": False,
    },
    "max_workers": 4,
    "target_language": "简体中文",
    "demucs": False,
    "vocal_separation_model": "htdemucs",
    "whisper": {
        "language": "en",
        "detected_language": "en",
        "runtime": "local",
        "backend": "qwen",
        "qwen_model": "1.7b",
        "qwen_engine": "auto",
        "model": "large-v3",
        "elevenlabs_api_key": "",
        "mai_api_key": "",
        "mai_region": "",
        "mai_provider": "azure",
        "mai_openrouter_api_key": "",
        "deepgram_api_key": "",
        "deepgram_model": "nova-3",
        "assembly_api_key": "",
        "assemblyai_lang": "en",
        "assemblyai_model": "universal",
        "whisperzero_api_key": "",
        "replicate_api_key": "",
    },
    "burn_subtitles": True,
    "ffmpeg_gpu": False,
    "youtube": {
        "cookies_path": "",
        "proxy": None,
    },
    "ytb_resolution": "1080",
    "subtitle": {
        "max_length": 75,
        "target_multiplier": 1.3,
        "sub_video_subtitles": "bilingual",
        "dub_video_subtitles": "translation",
        "style": {
            "source": {
                "font_name": "",
                "font_size": 15,
                "font_color": "&HFFFFFF",
                "outline_color": "&H000000",
                "outline_width": 1,
            },
            "translation": {
                "font_name": "",
                "font_size": 17,
                "font_color": "&H00FFFF",
                "outline_color": "&H000000",
                "outline_width": 1,
                "back_color": "&H33000000",
                "margin_v": 27,
            },
        },
    },
    "reflect_translate": True,
    "translation_style": "",
    "pause_before_translate": False,
    "pause_after_translate": False,
    "tts_method": "edge_tts",
    "sf_fish_tts": {
        "api_key": "",
        "voice": "anna",
        "custom_name": "",
        "voice_id": "",
        "mode": "preset",
    },
    "openai_tts": {
        "api_key": "",
        "voice": "alloy",
        "base_url": "https://api.openlux.ai/v1",
        "model": "gpt-4o-mini-tts",
    },
    "fish_audio": {
        "api_key": "",
        "mode": "clone",
        "voice_id": "74c6aba5cbf94a15bbdc547ffce5cb38",
        "voices": {
            "语彤 Yutong (中文, female)": "74c6aba5cbf94a15bbdc547ffce5cb38",
            "浩然 Haoran (中文, male)": "d675c275d1d44e57b4ef3840c5a23209",
        },
        "model": "s2.1-pro",
    },
    "sf_cosyvoice2": {
        "api_key": "",
    },
    "edge_tts": {
        "voice": "zh-CN-XiaoxiaoNeural",
    },
    "gpt_sovits": {
        "character": "Huanyuv2",
        "refer_mode": 3,
    },
    "f5tts": {
        "302_api": "",
    },
    "speed_factor": {
        "min": 1.0,
        "accept": 1.2,
        "max": 1.4,
    },
    "min_subtitle_duration": 2.5,
    "tolerance": 1.5,
    "audio": {
        "raw_sample_rate": 32000,
        "raw_bitrate": "128k",
    },
    "model_dir": "./_model_cache",
    "allowed_video_formats": ["mp4", "mov", "avi", "mkv", "flv", "wmv", "webm"],
    "allowed_audio_formats": ["wav", "mp3", "flac", "m4a"],
    "language_split_with_space": ["en", "es", "fr", "de", "it", "ru"],
    "language_split_without_space": ["zh", "ja"],
}


def is_placeholder_key(val: Any) -> bool:
    if not val or not isinstance(val, str):
        return True
    stripped = val.strip()
    if not stripped:
        return True
    return stripped.startswith(PLACEHOLDER_KEY_PREFIXES)


@contextlib.contextmanager
def task_config_context(overrides: dict[str, Any] | None):
    """Apply task-scoped configuration overrides without mutating global `config.yaml` (propagates to worker threads)."""
    global _PROCESS_CONFIG_OVERRIDES
    current = _TASK_CONFIG_OVERRIDES.get()
    if current is None:
        with _OVERRIDES_LOCK:
            current = _PROCESS_CONFIG_OVERRIDES or {}
    merged = {**current, **(overrides or {})}
    token = _TASK_CONFIG_OVERRIDES.set(merged)
    with _OVERRIDES_LOCK:
        prev_proc = _PROCESS_CONFIG_OVERRIDES
        _PROCESS_CONFIG_OVERRIDES = merged
    try:
        yield merged
    finally:
        with _OVERRIDES_LOCK:
            _PROCESS_CONFIG_OVERRIDES = prev_proc
        _TASK_CONFIG_OVERRIDES.reset(token)


def _atomic_dump_yaml(data: Any, config_path: str | Path) -> None:
    target = Path(config_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()


# -----------------------
# load & update config
# -----------------------

def load_key(key: str) -> Any:
    overrides = _TASK_CONFIG_OVERRIDES.get()
    if overrides is None:
        with _OVERRIDES_LOCK:
            overrides = _PROCESS_CONFIG_OVERRIDES
    if overrides:
        if key in overrides:
            return copy.deepcopy(overrides[key])
        prefix = f"{key}."
        sub_overrides = {k[len(prefix):]: v for k, v in overrides.items() if k.startswith(prefix)}
    else:
        sub_overrides = {}

    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

    if not isinstance(data, dict):
        raise KeyError(f"Configuration file {CONFIG_PATH!r} is empty or invalid")

    keys = key.split('.')
    value: Any = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Key '{k}' not found in configuration")

    if sub_overrides and isinstance(value, dict):
        merged = copy.deepcopy(dict(value))
        for sub_k, sub_v in sub_overrides.items():
            parts = sub_k.split(".")
            cur = merged
            for p in parts[:-1]:
                if not isinstance(cur.get(p), dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = sub_v
        return merged
    return value


def load_key_or(key, default=None):
    """Read an optional key; config.yaml files written before the key existed keep working."""
    try:
        return load_key(key)
    except KeyError:
        return default


def update_key(key: str, new_value: Any, add_missing: bool = False) -> bool:
    """Atomically set an existing key in `config.yaml`; `add_missing=True` also creates missing parent/leaf keys."""
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)
        if data is None:
            data = {}

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                if current[k] is None and add_missing:
                    current[k] = {}
                current = current[k]
            elif isinstance(current, dict) and add_missing:
                current[k] = {}
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and (keys[-1] in current or add_missing):
            current[keys[-1]] = new_value
            # Keep manual source-language changes atomic for UI and CLI callers.
            if key == "whisper.language" and new_value != "auto":
                current["detected_language"] = new_value
            _atomic_dump_yaml(data, CONFIG_PATH)
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")


def update_keys(updates: dict[str, Any], add_missing: bool = True) -> bool:
    """Atomically apply multiple dot-key updates to `config.yaml` in a single write."""
    if not updates:
        return True
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)
        if data is None:
            data = {}

        for key, new_value in updates.items():
            keys = key.split('.')
            current = data
            for k in keys[:-1]:
                if isinstance(current, dict) and k in current and isinstance(current[k], dict):
                    current = current[k]
                elif isinstance(current, dict) and add_missing:
                    current[k] = {}
                    current = current[k]
                else:
                    raise KeyError(f"Key '{k}' not found in configuration")
            if isinstance(current, dict) and (keys[-1] in current or add_missing):
                current[keys[-1]] = new_value
                if key == "whisper.language" and new_value != "auto":
                    current["detected_language"] = new_value
            else:
                raise KeyError(f"Key '{keys[-1]}' not found in configuration")

        _atomic_dump_yaml(data, CONFIG_PATH)
        return True


def ensure_config_schema(config_path: str | Path | None = None) -> bool:
    """Backfill any missing keys from `DEFAULT_CONFIG` into `config.yaml` while preserving comments."""
    target = Path(config_path if config_path is not None else CONFIG_PATH)
    if not target.is_file():
        with lock:
            _atomic_dump_yaml(copy.deepcopy(DEFAULT_CONFIG), target)
        return True

    with lock:
        with open(target, "r", encoding="utf-8") as f:
            data = yaml.load(f)
        if not isinstance(data, dict):
            data = {}

        changed = False

        def _merge_defaults(dst: dict[str, Any], src: dict[str, Any]) -> None:
            nonlocal changed
            for k, v in src.items():
                if k not in dst:
                    dst[k] = copy.deepcopy(v)
                    changed = True
                elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                    _merge_defaults(dst[k], v)

        _merge_defaults(data, DEFAULT_CONFIG)
        if changed:
            _atomic_dump_yaml(data, target)
        return changed


# basic utils
def get_source_language() -> str:
    whisper = load_key("whisper")
    language = whisper["language"]
    if language == "auto":
        language = whisper.get("detected_language")
    if not isinstance(language, str) or not language or language == "auto":
        raise ValueError("Source language is unknown. Run transcription first or select a language.")
    return language


# Written without spaces between words. Any other language (everything else Qwen3-ASR or
# WhisperX can detect, e.g. ko, vi, pt, ar) joins with a space unless config.yaml says otherwise.
LANGUAGES_WITHOUT_SPACE = {"zh", "yue", "ja", "th", "lo", "km", "my", "bo"}


def get_joiner(language: str) -> str:
    if language in load_key_or('language_split_with_space', []):
        return " "
    elif language in load_key_or('language_split_without_space', []):
        return ""
    if not isinstance(language, str) or not language or language == "auto":
        raise ValueError(f"Unsupported language code: {language!r}")
    return "" if language.lower() in LANGUAGES_WITHOUT_SPACE else " "


# Scripts that are written without spaces; their letters never get a space from join_words.
_UNSPACED_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HALFWIDTH KATAKANA", "IDEOGRAPHIC", "BOPOMOFO",
                     "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN")


def _spaced_word_char(char: str) -> bool:
    """A letter or digit of a script written with spaces (Latin, Cyrillic, Hangul, digits, ...)."""
    if not char.isalnum():
        return False
    try:
        return not unicodedata.name(char).startswith(_UNSPACED_SCRIPTS)
    except ValueError:
        return True


def get_text_joiner(text: str) -> str:
    """Joiner for text whose language code is unknown (translations): follow its script."""
    letters = [char for char in str(text) if char.isalpha()]
    unspaced = sum(1 for char in letters if not _spaced_word_char(char))
    return "" if unspaced * 2 > len(letters) else " "


def join_words(words: list[Any], joiner: str) -> str:
    """Join words/tokens with the language joiner, keeping code-switched words apart."""
    words = [str(word) for word in words]
    if joiner:
        return joiner.join(words)
    text = ""
    for word in words:
        if text and word:
            left, right = text[-1], word[0]
            if (not (left.isdecimal() and right.isdecimal())
                    and _spaced_word_char(right)
                    and (_spaced_word_char(left) or (left in ",.!?;:" and right.isalpha()))):
                text += " "
        text += word
    return text


if __name__ == "__main__":
    print(load_key('language_split_with_space'))

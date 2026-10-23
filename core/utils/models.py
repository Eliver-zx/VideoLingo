from __future__ import annotations

from pathlib import Path
from core.workspace import get_active_workspace

# ------------------------------------------
# 3 Core JSON Contracts (New Architecture)
# ------------------------------------------

_2_WORDS_JSON = "output/log/words.json"
_3_UTTERANCES_JSON = "output/log/utterances.json"
_4_1_TERMS_JSON = "output/log/terms.json"

# ------------------------------------------
# Legacy / Intermediate Path Constants
# ------------------------------------------

_2_CLEANED_CHUNKS = "output/log/cleaned_chunks.xlsx"
_3_1_SPLIT_BY_NLP = "output/log/split_by_nlp.txt"
_3_2_SPLIT_BY_MEANING = "output/log/split_by_meaning.txt"
_4_1_TERMINOLOGY = "output/log/terminology.json"
_4_2_TRANSLATION = "output/log/translation_results.xlsx"
_5_SPLIT_SUB = "output/log/translation_results_for_subtitles.xlsx"
_5_REMERGED = "output/log/translation_results_remerged.xlsx"

_8_1_AUDIO_TASK = "output/audio/tts_tasks.xlsx"
_10_DUB_TRUNCATED = "output/log/dub_truncated.json"


# ------------------------------------------
# Audio & Media Paths
# ------------------------------------------
_OUTPUT_DIR = "output"
_AUDIO_DIR = "output/audio"
_RAW_AUDIO_FILE = "output/audio/raw.mp3"
_VOCAL_AUDIO_FILE = "output/audio/vocal.mp3"
_BACKGROUND_AUDIO_FILE = "output/audio/background.mp3"
_DUB_WAV_FILE = "output/audio/dub.wav"
_BI_SRT = "output/bi.srt"
_BI_ASS = "output/bi.ass"
_6_BI_ASS = _BI_ASS
_AUDIO_REFERS_DIR = "output/audio/refers"
_AUDIO_SEGS_DIR = "output/audio/segs"
_AUDIO_TMP_DIR = "output/audio/tmp"

# ------------------------------------------
# Done markers
# ------------------------------------------
_TEXT_DONE_MARKER = "output/.subtitle_done"
_AUDIO_DONE_MARKER = "output/.dubbing_done"


def resolve_task_path(path_str: str | Path) -> str:
    """Resolve a default `output/...` path into the active `TaskWorkspace` while honoring test monkeypatches."""
    s = str(path_str).replace("\\", "/")
    ws = get_active_workspace()
    ws_root = str(ws.root).replace("\\", "/")
    if ws_root == "output":
        return s
    if s == "output":
        return ws_root
    if s.startswith("output/"):
        return f"{ws_root}/{s[len('output/'):]}"
    return s


__all__ = [
    "_2_WORDS_JSON",
    "_3_UTTERANCES_JSON",
    "_4_1_TERMS_JSON",
    "_2_CLEANED_CHUNKS",
    "_3_1_SPLIT_BY_NLP",
    "_3_2_SPLIT_BY_MEANING",
    "_4_1_TERMINOLOGY",
    "_4_2_TRANSLATION",
    "_5_SPLIT_SUB",
    "_5_REMERGED",
    "_8_1_AUDIO_TASK",
    "_10_DUB_TRUNCATED",
    "_OUTPUT_DIR",
    "_AUDIO_DIR",
    "_RAW_AUDIO_FILE",
    "_VOCAL_AUDIO_FILE",
    "_BACKGROUND_AUDIO_FILE",
    "_DUB_WAV_FILE",
    "_BI_SRT",
    "_BI_ASS",
    "_6_BI_ASS",
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR",
    "_TEXT_DONE_MARKER",
    "_AUDIO_DONE_MARKER",
    "resolve_task_path",
]

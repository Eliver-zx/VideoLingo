"""Cohesive subtitle formatting, timestamp alignment, and SRT sync helpers for Stage 6 (`core._6_gen_sub`)."""

from core.subtitle_gen.formatter import (
    _strip_quotes_and_dashes,
    clean_translation,
    convert_to_srt_format,
    format_subtitle_text,
    generate_subtitle_string,
    remove_punctuation,
)
from core.subtitle_gen.srt_sync import (
    _parse_srt_blocks,
    sync_subtitles_from_srt,
    sync_subtitles_from_srt_if_newer,
)
from core.subtitle_gen.timestamp_align import (
    _levenshtein_fast,
    _try_direct_utterance_timestamps,
    align_timestamp,
    get_sentence_timestamps,
    show_difference,
)

__all__ = [
    "_levenshtein_fast",
    "_parse_srt_blocks",
    "_strip_quotes_and_dashes",
    "_try_direct_utterance_timestamps",
    "align_timestamp",
    "clean_translation",
    "convert_to_srt_format",
    "format_subtitle_text",
    "generate_subtitle_string",
    "get_sentence_timestamps",
    "remove_punctuation",
    "show_difference",
    "sync_subtitles_from_srt",
    "sync_subtitles_from_srt_if_newer",
]

from __future__ import annotations

import os

import pandas as pd

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
    sync_subtitles_from_srt as _sync_subtitles_from_srt_impl,
    sync_subtitles_from_srt_if_newer as _sync_subtitles_from_srt_if_newer_impl,
)
from core.subtitle_gen.timestamp_align import (
    _levenshtein_fast,
    _try_direct_utterance_timestamps,
    align_timestamp,
    get_sentence_timestamps,
    show_difference,
)
from core.utils import get_joiner, load_key, load_key_or, log_event, report_progress  # noqa: F401
from core.utils.models import _OUTPUT_DIR, resolve_task_path
from core.workspace import get_active_workspace

SUBTITLE_OUTPUT_CONFIGS: list[tuple[str, list[str]]] = [
    ("src.srt", ["Source"]),
    ("trans.srt", ["Translation"]),
    ("src_trans.srt", ["Source", "Translation"]),
    ("trans_src.srt", ["Translation", "Source"]),
]


def gen_source_subtitles() -> None:
    """Transcription only: `src.srt` from the split sentences, without a translation."""
    from core._1_ytdlp import find_subtitle_file
    from core._5_split_sub import split_source_lines

    out_dir = resolve_task_path(_OUTPUT_DIR)
    ws = get_active_workspace()

    words = ws.load_words()
    df_text = (
        pd.DataFrame(words) if words else pd.DataFrame(columns=["text", "start", "end"])
    )

    utts = ws.load_utterances()
    lines = [
        str(u.get("src") or "").strip() for u in utts if str(u.get("src") or "").strip()
    ]

    # The subtitles of the user keep their lines
    df_source = pd.DataFrame(
        {"Source": lines if find_subtitle_file() else split_source_lines(lines)}
    )
    align_timestamp(
        df_text, df_source, [("src.srt", ["Source"])], out_dir, for_display=False
    )
    src_file = os.path.join(out_dir, "src.srt")
    if os.path.isfile(src_file):
        ws.set_kv("srt_synced_mtime", os.path.getmtime(src_file))
    ws.notify_artifact_update(
        "subtitles",
        message=f"已生成原文 SRT 字幕 ({len(df_source)} 条)",
        tag="llm",
        label="SUBS",
    )
    report_progress(
        1.0,
        f"已生成 {len(df_source)} 条原文字幕",
        f"原文 SRT 字幕生成完成 ({out_dir}/src.srt)",
        tag="llm",
        label="SUBS",
    )


def align_timestamp_main() -> None:
    out_dir = resolve_task_path(_OUTPUT_DIR)
    ws = get_active_workspace()

    words = ws.load_words()
    df_text = (
        pd.DataFrame(words) if words else pd.DataFrame(columns=["text", "start", "end"])
    )

    utts = ws.load_utterances()
    cue_rows: list[dict[str, str]] = []
    for u in utts:
        cues = u.get("sub_cues")
        if isinstance(cues, list) and cues:
            for c in cues:
                cue_rows.append(
                    {
                        "Source": str(c.get("src") or ""),
                        "Translation": str(
                            c.get("tgt", c.get("trans", c.get("text", ""))) or ""
                        ),
                    }
                )
        else:
            cue_rows.append(
                {
                    "Source": str(u.get("src") or ""),
                    "Translation": str(u.get("tgt") or ""),
                }
            )
    df_translate = pd.DataFrame(cue_rows, columns=["Source", "Translation"])
    df_translate["Translation"] = df_translate["Translation"].apply(clean_translation)

    align_timestamp(
        df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, out_dir, for_display=True
    )
    srt_mtimes = [
        os.path.getmtime(os.path.join(out_dir, fn))
        for fn in ("src.srt", "trans.srt")
        if os.path.isfile(os.path.join(out_dir, fn))
    ]
    if srt_mtimes:
        ws.set_kv("srt_synced_mtime", max(srt_mtimes))

    ws.notify_artifact_update(
        "subtitles",
        message=f"已生成双语 SRT 字幕文件 ({len(df_translate)} 条)",
        tag="llm",
        label="SUBS",
    )
    report_progress(
        1.0,
        f"已生成 {len(df_translate)} 条双语字幕",
        f"双语字幕已写入 ({out_dir})",
        tag="llm",
        label="SUBS",
    )


def sync_subtitles_from_srt() -> dict[str, int]:
    """Sync edited `src.srt` and `trans.srt` from disk back into `project.sqlite` and bilingual SRT files."""
    return _sync_subtitles_from_srt_impl(
        out_dir=resolve_task_path(_OUTPUT_DIR),
        ws=get_active_workspace(),
    )


def sync_subtitles_from_srt_if_newer() -> bool:
    """Auto-sync `src.srt` / `trans.srt` into SQLite if modified on disk after the last DB/SRT save."""
    return _sync_subtitles_from_srt_if_newer_impl(
        out_dir=resolve_task_path(_OUTPUT_DIR),
        ws=get_active_workspace(),
    )


__all__ = [
    "SUBTITLE_OUTPUT_CONFIGS",
    "_levenshtein_fast",
    "_parse_srt_blocks",
    "_strip_quotes_and_dashes",
    "_try_direct_utterance_timestamps",
    "align_timestamp",
    "align_timestamp_main",
    "clean_translation",
    "convert_to_srt_format",
    "format_subtitle_text",
    "gen_source_subtitles",
    "generate_subtitle_string",
    "get_sentence_timestamps",
    "remove_punctuation",
    "show_difference",
    "sync_subtitles_from_srt",
    "sync_subtitles_from_srt_if_newer",
]


if __name__ == "__main__":
    align_timestamp_main()

from __future__ import annotations

from typing import List

from core._1_ytdlp import find_subtitle_file
from core._3_2_split_meaning import (
    BR_TAG,
    find_split_positions,
    split_sentence as _default_split_sentence,
)
from core.concurrency import ContextThreadPoolExecutor
from core.prompts import (
    get_align_prompt as _default_get_align_prompt,
    get_batch_split_prompt,
)
from core.subtitle_split.clause_dp import (
    _merge_tiny_clauses,
    _partition_clauses_dp,
    _subsplit_overlong_cjk_clause,
    _try_local_clause_split,
)
from core.subtitle_split.llm_batch import (
    _apply_br_split_to_sentence,
    _batch_split_sentences_llm,
    _call_split_sentence,
    run_split_align_pass,
)
from core.subtitle_split.metrics import _normalize_book_marks, calc_len
from core.subtitle_split.ratio_align import (
    _find_spaced_cut,
    align_source_by_ratio,
    align_subs,
    get_target_parts,
)
from core.subtitle_split.word_span import (
    _align_part_end_words,
    _clean_alnum,
    _partition_word_span_for_cues,
    build_attempt_line_gaps,
    is_sub_split_cache_valid,
    populate_utterance_sub_cues,
    remap_origin_indices,
)
from core.utils import (  # noqa: F401
    ask_gpt,
    check_cancel,
    except_handler,
    get_joiner,
    join_words,
    load_key,
    load_key_or,
    log_event,
    report_progress,
)
from core.workspace import get_active_workspace

split_sentence = _default_split_sentence
get_align_prompt = _default_get_align_prompt


def split_align_subs(
    src_lines: List[str],
    tr_lines: List[str],
    line_word_gaps: List[List[float] | None] | None = None,
):
    subtitle_set = load_key("subtitle")
    return run_split_align_pass(
        src_lines=src_lines,
        tr_lines=tr_lines,
        max_sub_length=float(subtitle_set["max_length"]),
        target_sub_multiplier=float(subtitle_set["target_multiplier"]),
        target_language=load_key_or("target_language", None),
        use_legacy_align=get_align_prompt is not _default_get_align_prompt,
        use_custom_split=split_sentence is not _default_split_sentence,
        line_word_gaps=line_word_gaps,
    )


def split_source_lines(lines: List[str]) -> List[str]:
    """Split the lines that are too long for a subtitle, for subtitles without a translation."""
    max_length = load_key("subtitle")["max_length"]

    def process(line: str) -> List[str]:
        if calc_len(line) <= max_length:
            return [line]
        parts = [
            part.strip()
            for part in split_sentence(line, num_parts=2).split("\n")
            if part.strip()
        ]
        return parts if len(parts) > 1 else [line]

    for _ in range(3):
        if all(calc_len(line) <= max_length for line in lines):
            break
        check_cancel()
        with ContextThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            lines = [part for parts in executor.map(process, lines) for part in parts]
    return lines


def split_for_sub_main() -> None:
    ws = get_active_workspace()

    try:
        subtitle_set = load_key("subtitle")
    except FileNotFoundError:
        subtitle_set = {"max_length": 75, "target_multiplier": 1.2}
    max_sub_length = float(subtitle_set["max_length"])
    target_sub_multiplier = float(subtitle_set["target_multiplier"])

    utterances = ws.load_utterances()
    if is_sub_split_cache_valid(ws, utterances, max_sub_length, target_sub_multiplier):
        report_progress(
            1.0,
            "复用已有字幕切分结果",
            "字幕显示切分已是最新，跳过重复切分",
            tag="llm",
            label="CACHE",
        )
        return

    report_progress(
        0.05,
        "正在切分显示字幕…",
        f"开始执行长字幕切分与对齐（共 {len(utterances)} 句）",
        tag="llm",
        label="SPLIT",
    )

    if not utterances:
        raise FileNotFoundError(
            f"No translated utterances found in workspace database: {ws.db_path}"
        )
    src = [str(u.get("src") or "") for u in utterances]
    trans = [_normalize_book_marks(str(u.get("tgt") or "")) for u in utterances]

    if find_subtitle_file():
        for u in utterances:
            u["sub_split_config"] = {
                "max_length": max_sub_length,
                "target_multiplier": target_sub_multiplier,
            }
            u["sub_cues"] = [
                {
                    "src": str(u.get("src") or ""),
                    "tgt": str(u.get("tgt") or ""),
                    "trans": str(u.get("tgt") or ""),
                    "word_span": list(u.get("word_span") or [0, 0]),
                    "start": float(u.get("start") or 0.0),
                    "end": float(u.get("end") or 0.0),
                }
            ]
        rev = ws.save_utterances(utterances)
        ws.set_kv("sub_split_rev", rev)
        report_progress(
            1.0,
            "保留外部字幕行结构",
            "外部导入字幕保持原始行结构",
            tag="llm",
            label="SPLIT",
        )
        return

    orig_src = src.copy()
    orig_remerged = trans.copy()
    split_src, split_trans = src.copy(), trans.copy()
    origin_indices = list(range(len(orig_src)))

    words = ws.load_words()
    orig_word_gaps: dict[int, list[float] | None] = {}

    for attempt in range(3):
        report_progress(
            0.15 + attempt * 0.25,
            f"字幕切分第 {attempt + 1}/3 轮",
            f"执行字幕切分第 {attempt + 1} 轮",
            tag="llm",
            label="SPLIT",
        )
        prev_src = src.copy()
        line_gaps = build_attempt_line_gaps(
            utterances=utterances,
            orig_src=orig_src,
            words=words,
            src_lines=src,
            tr_lines=trans,
            origin_indices=origin_indices,
            max_sub_length=max_sub_length,
            target_sub_multiplier=target_sub_multiplier,
            gap_cache=orig_word_gaps,
        )
        split_src, split_trans, remerged = split_align_subs(
            src.copy(), trans, line_word_gaps=line_gaps
        )
        if attempt == 0:
            orig_remerged = remerged

        origin_indices = remap_origin_indices(origin_indices, prev_src, split_src)

        if all(
            calc_len(tr) * target_sub_multiplier <= max_sub_length for tr in split_trans
        ) and (
            len(split_src) == len(prev_src)
            or all(calc_len(s) <= max_sub_length for s in split_src)
        ):
            break

        src, trans = split_src, split_trans

    populate_utterance_sub_cues(
        utterances=utterances,
        orig_src=orig_src,
        orig_remerged=orig_remerged,
        origin_indices=origin_indices,
        split_src=split_src,
        split_trans=split_trans,
        words=words,
        max_sub_length=max_sub_length,
        target_sub_multiplier=target_sub_multiplier,
    )

    rev = ws.save_utterances(utterances)
    ws.set_kv("sub_split_rev", rev)
    report_progress(
        1.0,
        f"已生成 {len(split_src)} 条字幕行",
        f"字幕切分完成（{len(orig_src)} 句 -> {len(split_src)} 条字幕行）",
        tag="llm",
        label="SPLIT",
    )


__all__ = [
    "BR_TAG",
    "_align_part_end_words",
    "_apply_br_split_to_sentence",
    "_batch_split_sentences_llm",
    "_call_split_sentence",
    "_clean_alnum",
    "_default_get_align_prompt",
    "_default_split_sentence",
    "_find_spaced_cut",
    "_merge_tiny_clauses",
    "_normalize_book_marks",
    "_partition_clauses_dp",
    "_partition_word_span_for_cues",
    "_subsplit_overlong_cjk_clause",
    "_try_local_clause_split",
    "align_source_by_ratio",
    "align_subs",
    "ask_gpt",
    "calc_len",
    "find_split_positions",
    "find_subtitle_file",
    "get_align_prompt",
    "get_batch_split_prompt",
    "get_target_parts",
    "load_key",
    "load_key_or",
    "split_align_subs",
    "split_for_sub_main",
    "split_sentence",
    "split_source_lines",
]


if __name__ == "__main__":
    split_for_sub_main()

from __future__ import annotations

import inspect
import math
from typing import Any, List, Tuple

from core._3_2_split_meaning import BR_TAG, find_split_positions
from core.concurrency import ContextThreadPoolExecutor
from core.prompts import get_batch_split_prompt
from core.subtitle_split.clause_dp import _try_local_clause_split
from core.subtitle_split.metrics import (
    _CLAUSE_SPLIT_RE,
    _LEADING_PUNCT_RE,
    _normalize_book_marks,
    calc_len,
    needs_subtitle_split,
)
from core.subtitle_split.ratio_align import align_source_by_ratio, align_subs
from core.task_runner import StopTask
from core.utils import check_cancel, except_handler, log_event


def _apply_br_split_to_sentence(
    sentence: str, modified: str, language: str | None = None
) -> str:
    """Project ``[br]`` markers in ``modified`` back onto exact positions of ``sentence``."""
    mod_norm = BR_TAG.sub("[br]", str(modified or ""))
    parts = [p.strip() for p in mod_norm.split("[br]") if p.strip()]
    if len(parts) < 2:
        return sentence
    split_points = find_split_positions(sentence, "[br]".join(parts), language=language)
    if not split_points:
        return sentence
    best_split = sentence
    for idx_pt, split_point in enumerate(split_points):
        if idx_pt == 0:
            best_split = sentence[:split_point] + "\n" + sentence[split_point:]
        else:
            cur_parts = best_split.split("\n")
            last_part = cur_parts[-1]
            rel = split_point - split_points[idx_pt - 1]
            cur_parts[-1] = last_part[:rel] + "\n" + last_part[rel:]
            best_split = "\n".join(cur_parts)
    out_parts = [p.strip() for p in best_split.split("\n") if p.strip()]
    return "\n".join(out_parts) if len(out_parts) >= 2 else sentence


def _call_split_sentence(
    text: str, num_parts: int = 2, language: str | None = None
) -> str:
    import core._5_split_sub as _s5

    split_fn = _s5.split_sentence
    if language is not None:
        try:
            sig = inspect.signature(split_fn)
            if "language" in sig.parameters:
                return str(split_fn(text, num_parts=num_parts, language=language) or "")
        except (TypeError, ValueError):
            pass
    return str(split_fn(text, num_parts=num_parts) or "")


def _batch_split_sentences_llm(
    items: list[tuple[int, str, int]],
    language: str | None = None,
    batch_size: int = 12,
) -> dict[int, str]:
    """Split multiple punctuation-scarce sentences in parallel batches using a single LLM call per batch."""
    if not items:
        return {}

    import core._5_split_sub as _s5

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results: dict[int, str] = {}

    def _run_batch(
        batch: list[tuple[int, str, int]],
    ) -> tuple[dict[int, str], list[tuple[int, str, int]]]:
        payload = {
            str(line_idx): {"text": text, "num_parts": num_parts}
            for line_idx, text, num_parts in batch
        }
        prompt = get_batch_split_prompt(payload, language=language)

        def _valid_batch(resp: Any) -> dict[str, str]:
            if not isinstance(resp, dict):
                return {"status": "error", "message": "Response must be a JSON object"}
            valid_count = 0
            for k in payload:
                val = str(resp.get(k, ""))
                norm = BR_TAG.sub("[br]", val)
                parts = [p.strip() for p in norm.split("[br]") if p.strip()]
                if len(parts) >= 2:
                    valid_count += 1
            if valid_count == 0:
                return {
                    "status": "error",
                    "message": "No keys in batch response contained valid [br]-separated parts",
                }
            return {"status": "success", "message": "Batch split completed"}

        out: dict[int, str] = {}
        fallbacks: list[tuple[int, str, int]] = []
        try:
            resp = _s5.ask_gpt(
                prompt,
                resp_type="json",
                valid_def=_valid_batch,
                log_title="split_by_meaning_batch",
            )
            if isinstance(resp, dict):
                for line_idx, text, num_parts in batch:
                    raw_val = str(resp.get(str(line_idx), ""))
                    projected = _apply_br_split_to_sentence(
                        text, raw_val, language=language
                    )
                    if len([p for p in projected.split("\n") if p.strip()]) >= 2:
                        out[line_idx] = projected
                    else:
                        fallbacks.append((line_idx, text, num_parts))
                return out, fallbacks
        except StopTask:
            raise
        except Exception as exc:
            log_event(
                f"批量长句切分失败 ({exc})，回退至单句切分",
                tag="warn",
                label="SPLIT",
                level="warn",
            )

        return out, list(batch)

    def _run_single_fallback(item: tuple[int, str, int]) -> tuple[int, str]:
        line_idx, text, num_parts = item
        return (
            line_idx,
            _call_split_sentence(text, num_parts=num_parts, language=language).strip(),
        )

    max_workers = max(1, int(_s5.load_key_or("max_workers", 4) or 4))
    all_fallbacks: list[tuple[int, str, int]] = []
    with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_batch, b) for b in batches]
        for fut in futures:
            check_cancel()
            batch_out, batch_fallbacks = fut.result()
            results.update(batch_out)
            all_fallbacks.extend(batch_fallbacks)

        if all_fallbacks:
            if max_workers <= 1 or len(all_fallbacks) == 1:
                for fb_item in all_fallbacks:
                    check_cancel()
                    line_idx, split_val = _run_single_fallback(fb_item)
                    results[line_idx] = split_val
            else:
                fb_futures = [
                    executor.submit(_run_single_fallback, fb_item)
                    for fb_item in all_fallbacks
                ]
                for fb_fut in fb_futures:
                    check_cancel()
                    line_idx, split_val = fb_fut.result()
                    results[line_idx] = split_val

    return results


def run_split_align_pass(
    src_lines: List[str],
    tr_lines: List[str],
    max_sub_length: float,
    target_sub_multiplier: float,
    target_language: str | None,
    use_legacy_align: bool,
    use_custom_split: bool,
    line_word_gaps: List[List[float] | None] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    import core._5_split_sub as _s5

    remerged_tr_lines = tr_lines.copy()
    src_work: list[str | list[str]] = list(src_lines)
    tr_work: list[str | list[str]] = list(tr_lines)

    to_split: list[int] = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        if needs_subtitle_split(
            str(src), str(tr), float(max_sub_length), float(target_sub_multiplier)
        )[0]:
            to_split.append(i)

    if not use_legacy_align and not use_custom_split:
        llm_pending: list[tuple[int, str, int]] = []
        for i in to_split:
            gaps_i = (
                line_word_gaps[i]
                if (line_word_gaps and i < len(line_word_gaps))
                else None
            )
            local_res = _try_local_clause_split(
                str(src_lines[i]),
                str(tr_lines[i]),
                float(max_sub_length),
                float(target_sub_multiplier),
                word_gaps=gaps_i,
            )
            if local_res is not None:
                s_parts, t_parts, tr_remerged = local_res
                src_work[i] = s_parts
                tr_work[i] = t_parts
                remerged_tr_lines[i] = tr_remerged
            else:
                tr_norm = _normalize_book_marks(tr_lines[i])
                tr_clean = _LEADING_PUNCT_RE.sub("", tr_norm).strip() or tr_norm.strip()
                tr_w = calc_len(tr_clean) * float(target_sub_multiplier)
                raw_clauses_i = [
                    c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_clean) if c.strip()
                ]
                single_line_limit = min(
                    float(max_sub_length), max(45.0, float(max_sub_length) * 0.72)
                )
                if tr_w <= single_line_limit and all(
                    calc_len(c) * float(target_sub_multiplier) <= single_line_limit
                    for c in raw_clauses_i
                ):
                    continue
                needed_k = max(math.ceil(tr_w / max(1.0, single_line_limit)), 2)
                llm_pending.append((i, tr_clean, needed_k))

        if llm_pending:
            batch_splits = _batch_split_sentences_llm(
                llm_pending, language=target_language
            )
            for i, _tr_clean, _needed_k in llm_pending:
                split_tr = (batch_splits.get(i) or "").strip()
                if len([part for part in split_tr.split("\n") if part.strip()]) < 2:
                    log_event(
                        f"第 {i + 1} 行长句未能切分，保持原句",
                        tag="warn",
                        label="SPLIT",
                        level="warn",
                    )
                    continue
                gaps_i = (
                    line_word_gaps[i]
                    if (line_word_gaps and i < len(line_word_gaps))
                    else None
                )
                src_parts, tr_parts, tr_remerged = align_source_by_ratio(
                    str(src_lines[i]), str(tr_lines[i]), split_tr, word_gaps=gaps_i
                )
                if (
                    not tr_parts
                    or len(src_parts) < 2
                    or len(src_parts) != len(tr_parts)
                ):
                    log_event(
                        f"第 {i + 1} 行双语切分段数不一致，保持原句",
                        tag="warn",
                        label="SPLIT",
                        level="warn",
                    )
                    continue
                src_work[i] = src_parts
                tr_work[i] = tr_parts
                remerged_tr_lines[i] = tr_remerged
    else:

        @except_handler("Error in split_align_subs")
        def process(i: int) -> None:
            if use_legacy_align:
                split_src = _call_split_sentence(src_lines[i], num_parts=2).strip()
                if len([part for part in split_src.split("\n") if part.strip()]) < 2:
                    log_event(
                        f"第 {i + 1} 行长句未能切分，保持原句",
                        tag="warn",
                        label="SPLIT",
                        level="warn",
                    )
                    return
                src_parts, tr_parts, tr_remerged = align_subs(
                    src_lines[i], tr_lines[i], split_src
                )
            else:
                tr_norm = _normalize_book_marks(tr_lines[i])
                tr_clean = _LEADING_PUNCT_RE.sub("", tr_norm).strip() or tr_norm.strip()
                split_tr = _call_split_sentence(
                    tr_clean, num_parts=2, language=target_language
                ).strip()
                if len([part for part in split_tr.split("\n") if part.strip()]) < 2:
                    log_event(
                        f"第 {i + 1} 行长句未能切分，保持原句",
                        tag="warn",
                        label="SPLIT",
                        level="warn",
                    )
                    return
                gaps_i = (
                    line_word_gaps[i]
                    if (line_word_gaps and i < len(line_word_gaps))
                    else None
                )
                src_parts, tr_parts, tr_remerged = align_source_by_ratio(
                    str(src_lines[i]), str(tr_lines[i]), split_tr, word_gaps=gaps_i
                )
            if not tr_parts or len(src_parts) < 2 or len(src_parts) != len(tr_parts):
                log_event(
                    f"第 {i + 1} 行双语切分段数不一致，保持原句",
                    tag="warn",
                    label="SPLIT",
                    level="warn",
                )
                return
            src_work[i] = src_parts
            tr_work[i] = tr_parts
            remerged_tr_lines[i] = tr_remerged

        with ContextThreadPoolExecutor(
            max_workers=_s5.load_key("max_workers")
        ) as executor:
            futures = [executor.submit(process, i) for i in to_split]
        for i, future in zip(to_split, futures):
            error = future.exception()
            if isinstance(error, StopTask):
                raise error
            if error:
                log_event(
                    f"第 {i + 1} 行长句切分异常，保持原句: {error}",
                    tag="warn",
                    label="SPLIT",
                    level="warn",
                )

    flat_src = [
        item
        for sublist in src_work
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    flat_tr = [
        item
        for sublist in tr_work
        for item in (sublist if isinstance(sublist, list) else [sublist])
    ]
    return flat_src, flat_tr, remerged_tr_lines

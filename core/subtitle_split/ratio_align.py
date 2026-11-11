from __future__ import annotations

import re
from typing import Any, List, Tuple

from core.subtitle_split.metrics import _BREAK_CONJUNCTIONS, _CLAUSE_PUNCT, calc_len
from core.utils import join_words
from core.utils.config_utils import get_text_joiner

_EDGE_NON_WORD_RE = re.compile(r"^[^\w]+|[^\w]+$")


def get_target_parts(response_data: Any, num_parts: int) -> List[str] | None:
    """The aligned target parts of a response, or None unless there are exactly `num_parts` non-empty ones."""
    align_data = response_data.get("align") if isinstance(response_data, dict) else None
    if not isinstance(align_data, list) or len(align_data) != num_parts:
        return None
    parts: List[str] = []
    for i, item in enumerate(align_data):
        part = item.get(f"target_part_{i + 1}") if isinstance(item, dict) else None
        if not isinstance(part, str) or not part.strip():
            return None
        parts.append(part.strip())
    return parts


def align_subs(
    src_sub: str, tr_sub: str, src_part: str
) -> Tuple[List[str], List[str], str]:
    import core._5_split_sub as _s5

    align_prompt = _s5.get_align_prompt(src_sub, tr_sub, src_part)
    src_parts = src_part.split("\n")

    def valid_align(response_data: Any) -> dict[str, str]:
        if "align" not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if get_target_parts(response_data, len(src_parts)) is None:
            return {
                "status": "error",
                "message": f"Align does not contain {len(src_parts)} non-empty parts as expected!",
            }
        return {"status": "success", "message": "Align completed"}

    parsed = _s5.ask_gpt(
        align_prompt, resp_type="json", valid_def=valid_align, log_title="align_subs"
    )
    tr_parts = get_target_parts(parsed, len(src_parts)) or [tr_sub]

    # The parts are in the target language, so the source-language joiner does not apply
    tr_remerged = str(join_words(tr_parts, get_text_joiner("".join(tr_parts))))

    return src_parts, tr_parts, tr_remerged


def _find_spaced_cut(
    words: List[str],
    weights_or_target: List[float] | float,
    start_idx: int = 0,
    end_idx: int | None = None,
    target_ratio: float | None = None,
    word_gaps: List[float] | None = None,
) -> int:
    """Find a phrase-aware and acoustic-pause-weighted word cut index in (start_idx, end_idx)."""
    if isinstance(weights_or_target, (int, float)):
        weights = [max(1.0, calc_len(w)) for w in words]
        if end_idx is None:
            end_idx = len(words)
        target_w = float(weights_or_target)
    else:
        weights = list(weights_or_target)
        if end_idx is None:
            end_idx = len(words)
        seg_weights = weights[start_idx:end_idx]
        total_w = sum(seg_weights) or float(max(1, end_idx - start_idx))
        ratio = 0.5 if target_ratio is None else float(target_ratio)
        target_w = ratio * total_w

    if end_idx - start_idx <= 1:
        return start_idx + 1

    cum = 0.0
    ideal_cut = start_idx + 1
    min_diff = float("inf")
    for idx in range(start_idx + 1, end_idx):
        cum += weights[idx - 1]
        diff = abs(cum - target_w)
        if diff < min_diff:
            min_diff = diff
            ideal_cut = idx

    window = 3
    low = max(start_idx + 1, ideal_cut - window)
    high = min(end_idx - 1, ideal_cut + window)

    best_cut = ideal_cut
    best_score = -float("inf")
    for cut in range(low, high + 1):
        prev_word = words[cut - 1]
        next_word = words[cut]
        if prev_word.endswith("-") or next_word.startswith("-"):
            continue
        dist_penalty = abs(cut - ideal_cut) * 0.35
        bonus = 0.0
        if prev_word and prev_word[-1] in _CLAUSE_PUNCT:
            bonus = 2.0
        else:
            clean_next = _EDGE_NON_WORD_RE.sub("", next_word).lower()
            if clean_next in _BREAK_CONJUNCTIONS:
                bonus = 1.1
        gap_bonus = 0.0
        if word_gaps is not None and 0 <= cut - 1 < len(word_gaps):
            gap = max(0.0, float(word_gaps[cut - 1]))
            gap_bonus = 0.35 * min(gap, 0.8)
        score = bonus + gap_bonus - dist_penalty
        if score > best_score:
            best_score = score
            best_cut = cut
    return best_cut


def align_source_by_ratio(
    src_sub: str,
    tr_sub: str,
    tr_split: str,
    word_gaps: List[float] | None = None,
) -> Tuple[List[str], List[str], str]:
    """Split source sentence locally according to target-language split parts, phrase boundaries, and acoustic pauses."""
    tr_parts = [p.strip() for p in tr_split.split("\n") if p.strip()]
    tr_remerged = str(tr_sub)
    if len(tr_parts) < 2:
        return [src_sub], [tr_sub], tr_remerged

    tr_lens = [max(1.0, calc_len(p)) for p in tr_parts]
    total_tr_len = sum(tr_lens)
    ratios = [length / total_tr_len for length in tr_lens]
    num_parts = len(tr_parts)

    src_joiner = get_text_joiner(src_sub)
    if src_joiner == " ":
        src_words = src_sub.split()
        if len(src_words) < num_parts:
            return [src_sub], [tr_sub], tr_remerged
        weights = [max(1.0, calc_len(w)) for w in src_words]
        src_parts = []
        start_idx = 0
        rem_ratio = 1.0
        for part_idx in range(num_parts - 1):
            parts_left = num_parts - part_idx
            max_end = len(src_words) - (parts_left - 1)
            rel_ratio = (
                ratios[part_idx] / rem_ratio if rem_ratio > 0 else 1.0 / parts_left
            )
            cut_idx = _find_spaced_cut(
                src_words,
                weights,
                start_idx,
                len(src_words),
                rel_ratio,
                word_gaps=word_gaps,
            )
            cut_idx = min(max(start_idx + 1, cut_idx), max_end)
            src_parts.append(" ".join(src_words[start_idx:cut_idx]))
            start_idx = cut_idx
            rem_ratio = max(1e-6, rem_ratio - ratios[part_idx])
        src_parts.append(" ".join(src_words[start_idx:]))
    else:
        src_chars = list(src_sub.strip())
        if len(src_chars) < num_parts:
            return [src_sub], [tr_sub], tr_remerged
        weights = [max(1.0, calc_len(c)) for c in src_chars]
        src_parts = []
        start_idx = 0
        rem_ratio = 1.0
        for part_idx in range(num_parts - 1):
            parts_left = num_parts - part_idx
            max_end = len(src_chars) - (parts_left - 1)
            rel_ratio = (
                ratios[part_idx] / rem_ratio if rem_ratio > 0 else 1.0 / parts_left
            )
            seg_w = sum(weights[start_idx:])
            target_w = rel_ratio * seg_w
            cum = 0.0
            ideal_cut = start_idx + 1
            min_diff = float("inf")
            for idx in range(start_idx + 1, max_end + 1):
                cum += weights[idx - 1]
                diff = abs(cum - target_w)
                if diff < min_diff:
                    min_diff = diff
                    ideal_cut = idx
            window = 4
            low = max(start_idx + 1, ideal_cut - window)
            high = min(max_end, ideal_cut + window)
            if word_gaps is not None:
                best_cut = ideal_cut
                best_score = -float("inf")
                for i in range(low, high + 1):
                    punct_bonus = 2.0 if src_chars[i - 1] in _CLAUSE_PUNCT else 0.0
                    g = (
                        max(0.0, float(word_gaps[i - 1]))
                        if i - 1 < len(word_gaps)
                        else 0.0
                    )
                    score = punct_bonus + 0.35 * min(g, 0.8) - abs(i - ideal_cut) * 0.35
                    if score > best_score:
                        best_score = score
                        best_cut = i
                cut_idx = best_cut
            else:
                punct_cuts = [
                    i for i in range(low, high + 1) if src_chars[i - 1] in _CLAUSE_PUNCT
                ]
                cut_idx = (
                    min(punct_cuts, key=lambda i: abs(i - ideal_cut))
                    if punct_cuts
                    else ideal_cut
                )
            src_parts.append("".join(src_chars[start_idx:cut_idx]).strip())
            start_idx = cut_idx
            rem_ratio = max(1e-6, rem_ratio - ratios[part_idx])
        src_parts.append("".join(src_chars[start_idx:]).strip())

    if any(not p.strip() for p in src_parts) or len(src_parts) != len(tr_parts):
        return [src_sub], [tr_sub], tr_remerged

    return src_parts, tr_parts, tr_remerged

from __future__ import annotations

import concurrent.futures
import inspect
import math
import os
from pathlib import Path
import re
from typing import Any, List, Tuple

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core._1_ytdlp import find_subtitle_file
from core._3_2_split_meaning import (
    BR_TAG,
    find_split_positions,
    split_sentence as _default_split_sentence,
)
from core.prompts import (
    get_align_prompt as _default_get_align_prompt,
    get_batch_split_prompt,
)
from core.task_runner import StopTask
from core.utils import (  # noqa: F401
    ask_gpt,
    check_cancel,
    except_handler,
    get_joiner,
    join_words,
    load_key,
    load_key_or,
    rprint,
)
from core.utils.config_utils import get_text_joiner
from core.utils.models import (  # noqa: F401
    _2_CLEANED_CHUNKS,
    _3_2_SPLIT_BY_MEANING,
    _4_2_TRANSLATION,
    _5_REMERGED,
    _5_SPLIT_SUB,
    resolve_task_path,
)
from core.workspace import TaskWorkspace, get_active_workspace

console = Console()

split_sentence = _default_split_sentence
get_align_prompt = _default_get_align_prompt

_LEADING_PUNCT_RE = re.compile(r"^[\s，。、“”‘’！!？?：:；;（）()【】\[\]《》<>…—\-·]+")
_CLAUSE_PUNCT = set(",;:，、；：—")
_BREAK_CONJUNCTIONS = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "since",
    "although",
    "though",
    "while",
    "when",
    "where",
    "which",
    "that",
    "who",
    "if",
    "unless",
    "until",
    "before",
    "after",
    "whether",
    "as",
    "yet",
    "nor",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "from",
    "by",
    "about",
}

_LAST_SPLIT_CONFIG_BY_PATH: dict[str, tuple[float, float]] = {}


def calc_len(text: str) -> float:
    text = str(text)

    def char_weight(char: str) -> float:
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1.0
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1.0

    return sum(char_weight(char) for char in text)


def get_target_parts(response_data, num_parts):
    """The aligned target parts of a response, or None unless there are exactly `num_parts` non-empty ones."""
    align_data = response_data.get("align") if isinstance(response_data, dict) else None
    if not isinstance(align_data, list) or len(align_data) != num_parts:
        return None
    parts = []
    for i, item in enumerate(align_data):
        part = item.get(f"target_part_{i + 1}") if isinstance(item, dict) else None
        if not isinstance(part, str) or not part.strip():
            return None
        parts.append(part.strip())
    return parts


def align_subs(
    src_sub: str, tr_sub: str, src_part: str
) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    src_parts = src_part.split("\n")

    def valid_align(response_data):
        if "align" not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if get_target_parts(response_data, len(src_parts)) is None:
            return {
                "status": "error",
                "message": f"Align does not contain {len(src_parts)} non-empty parts as expected!",
            }
        return {"status": "success", "message": "Align completed"}

    parsed = ask_gpt(
        align_prompt, resp_type="json", valid_def=valid_align, log_title="align_subs"
    )
    tr_parts = get_target_parts(parsed, len(src_parts)) or [tr_sub]

    # The parts are in the target language, so the source-language joiner does not apply
    tr_remerged = str(join_words(tr_parts, get_text_joiner("".join(tr_parts))))

    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)

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
        seg_weights = weights[start_idx:end_idx]
        total_w = sum(seg_weights) or float(max(1, end_idx - start_idx))
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
            clean_next = re.sub(r"^[^\w]+|[^\w]+$", "", next_word).lower()
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

    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)

    return src_parts, tr_parts, tr_remerged


_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,；;：:。！？!?—])")


def _partition_clauses_dp(clauses: list[str], k: int) -> list[str]:
    """Partition ``clauses`` into ``k`` contiguous groups minimizing the maximum group weight."""
    n = len(clauses)
    if k <= 1 or n <= 1:
        return ["".join(clauses).strip()]
    k = min(k, n)
    weights = [max(1.0, calc_len(c)) for c in clauses]
    prefix = [0.0]
    for w in weights:
        prefix.append(prefix[-1] + w)

    dp = [[float("inf")] * (k + 1) for _ in range(n + 1)]
    parent = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            for prev in range(j - 1, i):
                seg_w = prefix[i] - prefix[prev]
                cost = max(dp[prev][j - 1], seg_w)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = prev

    groups: list[str] = []
    curr = n
    for j in range(k, 0, -1):
        prev = parent[curr][j]
        groups.append("".join(clauses[prev:curr]).strip())
        curr = prev
    groups.reverse()
    return [g for g in groups if g]


def _merge_tiny_clauses(
    clauses: list[str], min_weighted_len: float = 13.0
) -> list[str]:
    """Merge short conjunction/discourse clauses (e.g. '所以，', '但遗憾的是，', '然后我就想，') into the
    adjacent clause so subtitle cues never become 2–6 character fragments."""
    if len(clauses) <= 1:
        return clauses
    merged: list[str] = []
    carry = ""
    for c in clauses:
        cur = (carry + c).strip() if carry else c.strip()
        if calc_len(cur) < min_weighted_len:
            carry = cur
        else:
            merged.append(cur)
            carry = ""
    if carry:
        if merged:
            merged[-1] = (merged[-1] + carry).strip()
        else:
            merged.append(carry)
    return merged


def _try_local_clause_split(
    src_line: str,
    tr_line: str,
    max_sub_length: float,
    target_sub_multiplier: float,
    word_gaps: List[float] | None = None,
) -> Tuple[List[str], List[str], str] | None:
    """Split ``tr_line`` and ``src_line`` locally in O(1) using internal clause punctuation when
    existing clause boundaries already satisfy subtitle length constraints without an LLM call."""
    tr_clean = _LEADING_PUNCT_RE.sub("", str(tr_line)).strip() or str(tr_line).strip()
    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_clean) if c.strip()]
    clauses = _merge_tiny_clauses(raw_clauses, min_weighted_len=13.0)
    if len(clauses) < 2:
        return None
    if any(calc_len(c) * target_sub_multiplier > max_sub_length for c in clauses):
        return None

    tr_w = calc_len(tr_clean) * target_sub_multiplier
    src_w = calc_len(str(src_line))
    # Target-language-driven k with comfortable reading cadence (~22 CJK chars per cue
    # when splitting long sentences, while never exceeding max_sub_length).
    comfortable_target_w = min(max_sub_length, max(48.0, max_sub_length * 0.72))
    needed_k = max(
        math.ceil(tr_w / max(1.0, max_sub_length)),
        math.ceil(tr_w / comfortable_target_w) if src_w > max_sub_length else 1,
        2,
    )
    needed_k = min(needed_k, len(clauses))

    for k_try in range(needed_k, len(clauses) + 1):
        cand_groups = _partition_clauses_dp(clauses, k_try)
        if len(cand_groups) < 2 or any(calc_len(g) < 13.0 for g in cand_groups):
            continue
        if any(
            calc_len(g) * target_sub_multiplier > max_sub_length for g in cand_groups
        ):
            continue
        s_parts, t_parts, remerged = align_source_by_ratio(
            str(src_line), str(tr_line), "\n".join(cand_groups), word_gaps=word_gaps
        )
        if len(s_parts) == len(cand_groups) and all(
            calc_len(tp) * target_sub_multiplier <= max_sub_length for tp in t_parts
        ):
            return s_parts, t_parts, remerged
    return None


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


def _batch_split_sentences_llm(
    items: list[tuple[int, str, int]],
    language: str | None = None,
    batch_size: int = 12,
) -> dict[int, str]:
    """Split multiple punctuation-scarce sentences in parallel batches using a single LLM call per batch."""
    if not items:
        return {}

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    results: dict[int, str] = {}

    def _run_batch(batch: list[tuple[int, str, int]]) -> dict[int, str]:
        payload = {
            str(line_idx): {"text": text, "num_parts": num_parts}
            for line_idx, text, num_parts in batch
        }
        prompt = get_batch_split_prompt(payload, language=language)

        def _valid_batch(resp: Any) -> dict[str, str]:
            if not isinstance(resp, dict):
                return {"status": "error", "message": "Response must be a JSON object"}
            for k in payload:
                val = str(resp.get(k, ""))
                norm = BR_TAG.sub("[br]", val)
                if "[br]" not in norm:
                    return {"status": "error", "message": f"Missing [br] in key {k}"}
            return {"status": "success", "message": "Batch split completed"}

        out: dict[int, str] = {}
        try:
            resp = ask_gpt(
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
                        out[line_idx] = _call_split_sentence(
                            text, num_parts=num_parts, language=language
                        ).strip()
                return out
        except StopTask:
            raise
        except Exception:
            pass

        for line_idx, text, num_parts in batch:
            out[line_idx] = _call_split_sentence(
                text, num_parts=num_parts, language=language
            ).strip()
        return out

    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_batch, b) for b in batches]
        for fut in futures:
            check_cancel()
            results.update(fut.result())
    return results


def _call_split_sentence(
    text: str, num_parts: int = 2, language: str | None = None
) -> str:
    if language is not None:
        try:
            sig = inspect.signature(split_sentence)
            if "language" in sig.parameters:
                return str(
                    split_sentence(text, num_parts=num_parts, language=language) or ""
                )
        except (TypeError, ValueError):
            pass
    return str(split_sentence(text, num_parts=num_parts) or "")


def split_align_subs(
    src_lines: List[str],
    tr_lines: List[str],
    line_word_gaps: List[List[float] | None] | None = None,
):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    target_language = load_key_or("target_language", None)
    use_legacy_align = get_align_prompt is not _default_get_align_prompt
    use_custom_split = split_sentence is not _default_split_sentence

    src_work: list[str | list[str]] = list(src_lines)
    tr_work: list[str | list[str]] = list(tr_lines)

    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if (
            calc_len(src) > MAX_SUB_LENGTH
            or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH
        ):
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)

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
                float(MAX_SUB_LENGTH),
                float(TARGET_SUB_MULTIPLIER),
                word_gaps=gaps_i,
            )
            if local_res is not None:
                s_parts, t_parts, tr_remerged = local_res
                src_work[i] = s_parts
                tr_work[i] = t_parts
                remerged_tr_lines[i] = tr_remerged
            else:
                tr_clean = (
                    _LEADING_PUNCT_RE.sub("", str(tr_lines[i])).strip()
                    or str(tr_lines[i]).strip()
                )
                tr_w = calc_len(tr_clean) * float(TARGET_SUB_MULTIPLIER)
                # When the translation already fits comfortably on a single subtitle line
                # (<= MAX_SUB_LENGTH) and no single internal clause exceeds MAX_SUB_LENGTH,
                # keep it intact rather than chopping a single clause in half just because
                # the English source line had conversational filler words.
                raw_clauses_i = [
                    c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_clean) if c.strip()
                ]
                if tr_w <= float(MAX_SUB_LENGTH) and all(
                    calc_len(c) * float(TARGET_SUB_MULTIPLIER) <= float(MAX_SUB_LENGTH)
                    for c in raw_clauses_i
                ):
                    continue
                needed_k = max(
                    math.ceil(tr_w / max(1.0, float(MAX_SUB_LENGTH))),
                    2,
                )
                llm_pending.append((i, tr_clean, needed_k))

        if llm_pending:
            batch_splits = _batch_split_sentences_llm(
                llm_pending, language=target_language
            )
            for i, _tr_clean, _needed_k in llm_pending:
                split_tr = (batch_splits.get(i) or "").strip()
                if len([part for part in split_tr.split("\n") if part.strip()]) < 2:
                    console.print(
                        f"[yellow]⚠️ Line {i} could not be split, keeping it as is[/yellow]"
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
                    console.print(
                        f"[yellow]⚠️ Line {i} was aligned into a different number of parts, keeping it as is[/yellow]"
                    )
                    continue
                src_work[i] = src_parts
                tr_work[i] = tr_parts
                remerged_tr_lines[i] = tr_remerged
    else:

        @except_handler("Error in split_align_subs")
        def process(i):
            if use_legacy_align:
                split_src = _call_split_sentence(src_lines[i], num_parts=2).strip()
                if len([part for part in split_src.split("\n") if part.strip()]) < 2:
                    console.print(
                        f"[yellow]⚠️ Line {i} could not be split, keeping it as is[/yellow]"
                    )
                    return
                src_parts, tr_parts, tr_remerged = align_subs(
                    src_lines[i], tr_lines[i], split_src
                )
            else:
                tr_clean = (
                    _LEADING_PUNCT_RE.sub("", str(tr_lines[i])).strip()
                    or str(tr_lines[i]).strip()
                )
                split_tr = _call_split_sentence(
                    tr_clean, num_parts=2, language=target_language
                ).strip()
                if len([part for part in split_tr.split("\n") if part.strip()]) < 2:
                    console.print(
                        f"[yellow]⚠️ Line {i} could not be split, keeping it as is[/yellow]"
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
            # Source and translation must stay in step: a line is split on both sides or not at all
            if not tr_parts or len(src_parts) < 2 or len(src_parts) != len(tr_parts):
                console.print(
                    f"[yellow]⚠️ Line {i} was aligned into a different number of parts, keeping it as is[/yellow]"
                )
                return
            src_work[i] = src_parts
            tr_work[i] = tr_parts
            remerged_tr_lines[i] = tr_remerged

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=load_key("max_workers")
        ) as executor:
            futures = [executor.submit(process, i) for i in to_split]
        for i, future in zip(to_split, futures):
            error = future.exception()
            if isinstance(error, StopTask):
                raise error
            if error:
                console.print(f"[yellow]⚠️ Line {i} is kept unsplit: {error}[/yellow]")

    # Flatten `src_work` and `tr_work`
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


def split_source_lines(lines: List[str]) -> List[str]:
    """Split the lines that are too long for a subtitle, for subtitles without a translation."""
    max_length = load_key("subtitle")["max_length"]

    def process(line):
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
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=load_key("max_workers")
        ) as executor:
            lines = [part for parts in executor.map(process, lines) for part in parts]
    return lines


def _clean_alnum(text: str) -> str:
    return re.sub(r"[^\w]", "", str(text).lower())


def _partition_word_span_for_cues(
    w_start: int,
    w_end: int,
    src_parts: list[str],
    tr_parts: list[str],
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Partition ``[w_start, w_end)`` into exact ``word_span: [w_a, w_b)`` cues with O(1) timestamps."""
    n_total = len(words)
    w_start = max(0, min(w_start, max(0, n_total - 1)))
    w_end = max(w_start + 1, min(w_end, n_total)) if n_total > 0 else 0
    if not src_parts or n_total == 0:
        return []
    if len(src_parts) == 1:
        return [
            {
                "src": src_parts[0],
                "tgt": tr_parts[0] if tr_parts else "",
                "word_span": [w_start, w_end],
                "start": round(float(words[w_start].get("start", 0.0)), 3),
                "end": round(float(words[w_end - 1].get("end", 0.0)), 3),
            }
        ]

    span_clean = ""
    char_to_abs_word: list[int] = []
    for abs_w in range(w_start, w_end):
        cw = _clean_alnum(words[abs_w].get("text", words[abs_w].get("word", "")))
        span_clean += cw
        char_to_abs_word.extend([abs_w] * len(cw))

    cues: list[dict[str, Any]] = []
    curr_w = w_start
    char_cursor = 0
    num_cues = len(src_parts)

    for idx, (s_part, t_part) in enumerate(zip(src_parts, tr_parts)):
        if idx == num_cues - 1:
            w_a = min(curr_w, w_end - 1)
            w_b = w_end
        else:
            part_clean = _clean_alnum(s_part)
            pos = (
                span_clean.find(part_clean, char_cursor)
                if (part_clean and span_clean)
                else -1
            )
            if pos != -1 and pos + len(part_clean) - 1 < len(char_to_abs_word):
                w_a = curr_w
                w_b = max(w_a + 1, char_to_abs_word[pos + len(part_clean) - 1] + 1)
                char_cursor = pos + len(part_clean)
            else:
                rem_words = max(1, w_end - curr_w)
                rem_cues = num_cues - idx
                step = max(1, round(rem_words / rem_cues))
                w_a = curr_w
                w_b = min(w_end - (rem_cues - 1), w_a + step)
                w_b = max(w_a + 1, w_b)
            curr_w = min(w_b, w_end - 1)

        cues.append(
            {
                "src": s_part,
                "tgt": t_part,
                "word_span": [w_a, w_b],
                "start": round(float(words[w_a].get("start", 0.0)), 3),
                "end": round(float(words[w_b - 1].get("end", 0.0)), 3),
            }
        )
    return cues


def split_for_sub_main():
    split_sub_path = resolve_task_path(_5_SPLIT_SUB)
    remerged_path = resolve_task_path(_5_REMERGED)
    trans_path = resolve_task_path(_4_2_TRANSLATION)
    abs_split_path = os.path.abspath(split_sub_path)

    ws = (
        get_active_workspace()
        if str(_5_SPLIT_SUB).startswith("output/")
        else TaskWorkspace(
            Path(split_sub_path).parent.parent
            if Path(split_sub_path).parent.name == "log"
            else Path(split_sub_path).parent
        )
    )

    try:
        subtitle_set = load_key("subtitle")
    except FileNotFoundError:
        subtitle_set = {"max_length": 75, "target_multiplier": 1.2}
    MAX_SUB_LENGTH = float(subtitle_set["max_length"])
    TARGET_SUB_MULTIPLIER = float(subtitle_set["target_multiplier"])

    if (
        os.path.exists(split_sub_path)
        and os.path.exists(remerged_path)
        and os.path.exists(trans_path)
        and os.path.getmtime(split_sub_path) >= os.path.getmtime(trans_path)
    ):
        try:
            prev_cfg = _LAST_SPLIT_CONFIG_BY_PATH.get(abs_split_path)
            if prev_cfg is None and ws.utterances_path.is_file():
                utters = ws.load_utterances()
                if utters and isinstance(utters[0].get("sub_split_config"), dict):
                    cfg_d = utters[0]["sub_split_config"]
                    prev_cfg = (
                        float(cfg_d.get("max_length", -1)),
                        float(cfg_d.get("target_multiplier", -1)),
                    )
            config_matches = prev_cfg is None or prev_cfg == (
                MAX_SUB_LENGTH,
                TARGET_SUB_MULTIPLIER,
            )
            df_existing = pd.read_excel(split_sub_path)
            if (
                config_matches
                and all(
                    calc_len(str(s)) <= MAX_SUB_LENGTH
                    for s in df_existing["Source"].fillna("")
                )
                and all(
                    calc_len(str(tr)) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH
                    for tr in df_existing["Translation"].fillna("")
                )
            ):
                rprint(
                    f"[yellow]⚠️ File <{split_sub_path}> already exists and is up-to-date, "
                    "skip <split_for_sub_main> step.[/yellow]"
                )
                return
        except Exception:
            pass

    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")

    df = pd.read_excel(trans_path)
    src = df["Source"].tolist()
    trans = df["Translation"].tolist()

    if find_subtitle_file():
        # The subtitles of the user keep their lines and their times
        console.print(
            "[cyan]📄 Subtitles of the user: the lines are kept as they are[/cyan]"
        )
        os.makedirs(os.path.dirname(split_sub_path) or ".", exist_ok=True)
        for path in (split_sub_path, remerged_path):
            pd.DataFrame({"Source": src, "Translation": trans}).to_excel(
                path, index=False
            )
        _LAST_SPLIT_CONFIG_BY_PATH[abs_split_path] = (
            MAX_SUB_LENGTH,
            TARGET_SUB_MULTIPLIER,
        )
        return

    orig_src = src.copy()
    orig_remerged = trans.copy()
    split_src, split_trans = src.copy(), trans.copy()
    origin_indices = list(range(len(orig_src)))

    for attempt in range(3):
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        prev_src = src.copy()
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)
        if attempt == 0:
            orig_remerged = remerged

        # Update origin_indices if split_align_subs expanded any lines
        if len(split_src) != len(prev_src):
            new_indices: list[int] = []
            cursor = 0
            for orig_idx, old_s in zip(origin_indices, prev_src):
                if cursor < len(split_src) and split_src[cursor] == old_s:
                    new_indices.append(orig_idx)
                    cursor += 1
                else:
                    # Gather parts until their combined cleaned length covers old_s
                    target_clean_len = len(_clean_alnum(old_s))
                    acc_len = 0
                    while cursor < len(split_src):
                        new_indices.append(orig_idx)
                        acc_len += len(_clean_alnum(split_src[cursor]))
                        cursor += 1
                        if acc_len >= target_clean_len:
                            break
            if len(new_indices) == len(split_src):
                origin_indices = new_indices

        if all(
            calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans
        ) and (
            len(split_src) == len(prev_src)
            or all(calc_len(s) <= MAX_SUB_LENGTH for s in split_src)
        ):
            break

        src, trans = split_src, split_trans

    os.makedirs(os.path.dirname(split_sub_path) or ".", exist_ok=True)
    pd.DataFrame({"Source": split_src, "Translation": split_trans}).to_excel(
        split_sub_path, index=False
    )
    pd.DataFrame({"Source": orig_src, "Translation": orig_remerged}).to_excel(
        remerged_path, index=False
    )
    _LAST_SPLIT_CONFIG_BY_PATH[abs_split_path] = (MAX_SUB_LENGTH, TARGET_SUB_MULTIPLIER)

    if ws.utterances_path.is_file() and ws.words_path.is_file():
        utterances = ws.load_utterances()
        words = ws.load_words()
        if len(utterances) == len(orig_src) and len(origin_indices) == len(split_src):
            grouped_src: dict[int, list[str]] = {i: [] for i in range(len(orig_src))}
            grouped_tr: dict[int, list[str]] = {i: [] for i in range(len(orig_src))}
            for u_idx, s_part, t_part in zip(origin_indices, split_src, split_trans):
                grouped_src[u_idx].append(str(s_part))
                grouped_tr[u_idx].append(str(t_part))
            for u_idx, u in enumerate(utterances):
                span = u.get("word_span") or [0, len(words)]
                w_a = int(span[0]) if len(span) >= 2 else 0
                w_b = int(span[1]) if len(span) >= 2 else len(words)
                u["sub_split_config"] = {
                    "max_length": MAX_SUB_LENGTH,
                    "target_multiplier": TARGET_SUB_MULTIPLIER,
                }
                u["sub_cues"] = _partition_word_span_for_cues(
                    w_a,
                    w_b,
                    grouped_src.get(u_idx) or [str(orig_src[u_idx])],
                    grouped_tr.get(u_idx) or [str(orig_remerged[u_idx])],
                    words,
                )
            ws.save_utterances(utterances)


if __name__ == "__main__":
    split_for_sub_main()

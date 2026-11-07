from __future__ import annotations

import difflib
import inspect
import math
import re
from typing import Any, List, Tuple

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
from core.task_runner import StopTask
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
from core.utils.config_utils import get_text_joiner
from core.workspace import get_active_workspace

split_sentence = _default_split_sentence
get_align_prompt = _default_get_align_prompt

_LEADING_PUNCT_RE = re.compile(
    r"^[\s，。、！!？?：:；;）)】\]》>…—\-·“”‘’\"'「」『』]+"
)
_QUOTE_BOOK_MARK_TABLE = str.maketrans({c: None for c in "《》〈〉“”‘’「」『』"})


def _normalize_book_marks(text: str) -> str:
    s = str(text).translate(_QUOTE_BOOK_MARK_TABLE)
    if re.search(r"[\u4e00-\u9fff]", s):
        s = s.replace('"', "")
        s = re.sub(r"(?:——+|—+|--+|[：；;])+", "，", s)
        s = re.sub(r"(?<!\d):(?!\d)", "，", s)
        s = re.sub(r"，+", "，", s)
        s = re.sub(r"，([。？！、])", r"\1", s)
        s = re.sub(r"([，。？！、])，", r"\1", s)
        s = re.sub(r"^，+|，+$", "", s)
    return s


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

    return src_parts, tr_parts, tr_remerged


_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,、；;：:。！？!?])|(?<=——)|(?<!—)(?<=—)(?!—)")


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
    clauses: list[str],
    min_weighted_len: float = 7.5,
    lead_in_weighted_len: float = 13.0,
    max_merged_weighted_len: float = 62.5,
) -> list[str]:
    """Merge tiny conjunction/discourse clauses into the adjacent clause:
    - Hard fragments (`calc_len < min_weighted_len`, ~1-3 CJK chars or 1 short word) always merge.
    - Sentence-initial lead-in clauses (`calc_len < lead_in_weighted_len`, ~4-6 CJK chars) merge
      into the second clause only when the combined clause still fits within `max_merged_weighted_len`,
      preserving mid-sentence parallel short clauses in deliberate speech."""
    if len(clauses) <= 1:
        return clauses
    merged: list[str] = []
    carry = ""
    for idx, c in enumerate(clauses):
        cur = (carry + c).strip() if carry else c.strip()
        cur_w = calc_len(cur)
        if cur_w < min_weighted_len:
            carry = cur
        elif (
            idx == 0
            and not merged
            and cur_w < lead_in_weighted_len
            and idx + 1 < len(clauses)
            and (cur_w + calc_len(clauses[idx + 1])) <= max_merged_weighted_len
        ):
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


_CJK_SYNTACTIC_SPLIT_RE = re.compile(
    r"(?:(?<=适用于)|(?<=取决于)|(?<=相当于)|(?<=得益于)|(?<=来自于)|(?<=是由于)|(?<=是因为)|(?<=指的是)|(?<=意味着)|(?<=的时候)|(?<=的情况下)|(?<=的同时)|(?<=之后)|(?<=之前)|(?<=的)|(?=而且)|(?=并且)|(?=同时)|(?=以及)|(?=或者)|(?=从而)|(?=因此)|(?=所以)|(?=因为)|(?=如果)|(?=所[\u4e00-\u9fff]{1,2}(?:的)?))"
)


def _subsplit_overlong_cjk_clause(
    clause: str, max_sub_length: float, target_sub_multiplier: float
) -> List[str]:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", clause))
    if cjk_count == 0 or (
        calc_len(clause) * target_sub_multiplier <= max_sub_length and cjk_count <= 25
    ):
        return [clause]
    max_single_w = min(float(max_sub_length), max(50.0, float(max_sub_length) * 0.76))
    n = len(clause)
    low_b = max(4, int(n * 0.25))
    high_b = min(n - 4, int(n * 0.75))
    mid = n // 2
    best_idx = -1
    best_dist = n
    for m in _CJK_SYNTACTIC_SPLIT_RE.finditer(clause):
        idx = m.start()
        if not (low_b <= idx <= high_b):
            continue
        left, right = clause[:idx].strip(), clause[idx:].strip()
        if (
            calc_len(left) >= 12.25
            and calc_len(right) >= 12.25
            and calc_len(left) * target_sub_multiplier <= max_single_w
            and calc_len(right) * target_sub_multiplier <= max_single_w
        ):
            dist = abs(idx - mid)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
    if best_idx != -1:
        return [clause[:best_idx].strip(), clause[best_idx:].strip()]
    return [clause]


def _try_local_clause_split(
    src_line: str,
    tr_line: str,
    max_sub_length: float,
    target_sub_multiplier: float,
    word_gaps: List[float] | None = None,
) -> Tuple[List[str], List[str], str] | None:
    """Split ``tr_line`` and ``src_line`` locally in O(1) using internal clause punctuation when
    existing clause boundaries already satisfy subtitle length constraints without an LLM call."""
    tr_norm = _normalize_book_marks(tr_line)
    tr_clean = _LEADING_PUNCT_RE.sub("", tr_norm).strip() or tr_norm.strip()
    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_clean) if c.strip()]
    if len(raw_clauses) >= 2:
        expanded: List[str] = []
        for rc in raw_clauses:
            expanded.extend(
                _subsplit_overlong_cjk_clause(rc, max_sub_length, target_sub_multiplier)
            )
        raw_clauses = expanded
    max_clause_w = max_sub_length / max(0.1, target_sub_multiplier)
    lead_merge_cap = min(max_clause_w, 35.0)
    clauses = _merge_tiny_clauses(
        raw_clauses,
        min_weighted_len=7.5,
        lead_in_weighted_len=13.0,
        max_merged_weighted_len=lead_merge_cap,
    )
    if len(clauses) < 2:
        return None
    if any(calc_len(c) * target_sub_multiplier > max_sub_length for c in clauses):
        return None

    tr_w = calc_len(tr_clean) * target_sub_multiplier
    comfortable_target_w = min(max_sub_length, max(38.0, max_sub_length * 0.60))
    needed_k = max(
        math.ceil(tr_w / max(1.0, max_sub_length)),
        math.ceil(tr_w / comfortable_target_w),
        2,
    )
    needed_k = min(needed_k, len(clauses))

    for k_try in range(needed_k, len(clauses) + 1):
        cand_groups = _partition_clauses_dp(clauses, k_try)
        if len(cand_groups) < 2 or any(calc_len(g) < 7.5 for g in cand_groups):
            continue
        if any(
            calc_len(g) * target_sub_multiplier > max_sub_length for g in cand_groups
        ):
            continue
        s_parts, t_parts, remerged = align_source_by_ratio(
            str(src_line), tr_norm, "\n".join(cand_groups), word_gaps=word_gaps
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
        except Exception as exc:
            log_event(
                f"批量长句切分失败 ({exc})，回退至单句切分",
                tag="warn",
                label="SPLIT",
                level="warn",
            )

        for line_idx, text, num_parts in batch:
            out[line_idx] = _call_split_sentence(
                text, num_parts=num_parts, language=language
            ).strip()
        return out

    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))
    with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
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

    comfortable_w = max(38.0, float(MAX_SUB_LENGTH) * 0.60)
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        tr_norm_i = _normalize_book_marks(tr)
        has_multi_clauses = (
            calc_len(tr_norm_i) * float(TARGET_SUB_MULTIPLIER) > comfortable_w
            and len([c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_norm_i) if c.strip()])
            >= 2
        )
        if (
            calc_len(src) > MAX_SUB_LENGTH
            or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH
            or has_multi_clauses
        ):
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
                tr_norm = _normalize_book_marks(tr_lines[i])
                tr_clean = _LEADING_PUNCT_RE.sub("", tr_norm).strip() or tr_norm.strip()
                tr_w = calc_len(tr_clean) * float(TARGET_SUB_MULTIPLIER)
                raw_clauses_i = [
                    c.strip() for c in _CLAUSE_SPLIT_RE.split(tr_clean) if c.strip()
                ]
                single_line_limit = min(
                    float(MAX_SUB_LENGTH), max(45.0, float(MAX_SUB_LENGTH) * 0.72)
                )
                if tr_w <= single_line_limit and all(
                    calc_len(c) * float(TARGET_SUB_MULTIPLIER) <= single_line_limit
                    for c in raw_clauses_i
                ):
                    continue
                needed_k = max(
                    math.ceil(tr_w / max(1.0, single_line_limit)),
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
        def process(i):
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
            # Source and translation must stay in step: a line is split on both sides or not at all
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

        with ContextThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
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
        with ContextThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            lines = [part for parts in executor.map(process, lines) for part in parts]
    return lines


def _clean_alnum(text: str) -> str:
    from core._3_1_split_nlp import _clean_word

    return _clean_word(text)


def _align_part_end_words(
    w_start: int,
    w_end: int,
    src_parts: list[str],
    words: list[dict[str, Any]],
) -> dict[int, int]:
    """Map each cue index `idx` to its inclusive ending absolute word index via token SequenceMatcher."""
    from core._3_1_split_nlp import _split_atomic_tokens

    span_toks: list[str] = []
    span_abs_w: list[int] = []
    for abs_w in range(w_start, w_end):
        raw_w = str(words[abs_w].get("text", words[abs_w].get("word", "")))
        for m in _split_atomic_tokens(raw_w):
            cw = _clean_alnum(m)
            if cw:
                span_toks.append(cw)
                span_abs_w.append(abs_w)

    part_toks: list[str] = []
    part_end_j: dict[int, int] = {}
    for idx, s_part in enumerate(src_parts):
        for m in _split_atomic_tokens(str(s_part)):
            cw = _clean_alnum(m)
            if cw:
                part_toks.append(cw)
        if part_toks:
            part_end_j[idx] = len(part_toks) - 1

    if not span_toks or not part_toks:
        return {}

    matcher = difflib.SequenceMatcher(None, span_toks, part_toks, autojunk=False)
    j_to_i: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(j2 - j1):
                j_to_i[j1 + off] = i1 + off
        elif tag == "replace" and i2 > i1 and j2 > j1:
            s_len = i2 - i1
            p_len = j2 - j1
            for off in range(p_len - 1):
                j_to_i[j1 + off] = i1 + min(
                    s_len - 1, max(0, int((off + 1) * s_len / p_len) - 1)
                )
            j_to_i[j2 - 1] = i2 - 1

    cue_end_abs_w: dict[int, int] = {}
    for idx, end_j in part_end_j.items():
        for probe_j in range(end_j, -1, -1):
            if probe_j in j_to_i:
                cue_end_abs_w[idx] = span_abs_w[j_to_i[probe_j]]
                break
    return cue_end_abs_w


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
    word_char_start: dict[int, int] = {}
    for abs_w in range(w_start, w_end):
        word_char_start[abs_w] = len(span_clean)
        cw = _clean_alnum(words[abs_w].get("text", words[abs_w].get("word", "")))
        span_clean += cw
        char_to_abs_word.extend([abs_w] * len(cw))
    word_char_start[w_end] = len(span_clean)

    aligned_ends: dict[int, int] | None = None
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
                if aligned_ends is None:
                    aligned_ends = _align_part_end_words(
                        w_start, w_end, src_parts, words
                    )
                if idx in aligned_ends and aligned_ends[idx] >= curr_w:
                    w_a = curr_w
                    max_allowed = max(w_a + 1, w_end - (num_cues - 1 - idx))
                    w_b = min(max_allowed, aligned_ends[idx] + 1)
                    w_b = max(w_a + 1, w_b)
                else:
                    rem_words = max(1, w_end - curr_w)
                    rem_cues = num_cues - idx
                    step = max(1, round(rem_words / rem_cues))
                    w_a = curr_w
                    w_b = min(w_end - (rem_cues - 1), w_a + step)
                    w_b = max(w_a + 1, w_b)
                char_cursor = word_char_start.get(w_b, len(span_clean))
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
    ws = get_active_workspace()

    try:
        subtitle_set = load_key("subtitle")
    except FileNotFoundError:
        subtitle_set = {"max_length": 75, "target_multiplier": 1.2}
    MAX_SUB_LENGTH = float(subtitle_set["max_length"])
    TARGET_SUB_MULTIPLIER = float(subtitle_set["target_multiplier"])

    utterances = ws.load_utterances()
    if utterances and ws.has_sub_cues():
        try:
            prev_cfg = None
            if isinstance(utterances[0].get("sub_split_config"), dict):
                cfg_d = utterances[0]["sub_split_config"]
                prev_cfg = (
                    float(cfg_d.get("max_length", -1)),
                    float(cfg_d.get("target_multiplier", -1)),
                )
            saved_rev = ws.get_kv("sub_split_rev")
            rev_matches = (
                saved_rev is not None and int(saved_rev) == ws.get_utterances_revision()
            )
            config_matches = prev_cfg == (MAX_SUB_LENGTH, TARGET_SUB_MULTIPLIER)
            all_cues = [
                c
                for u in utterances
                for c in (u.get("sub_cues") or [])
                if isinstance(c, dict)
            ]
            if (
                rev_matches
                and config_matches
                and all_cues
                and all(
                    calc_len(str(c.get("src") or "")) <= MAX_SUB_LENGTH
                    for c in all_cues
                )
                and all(
                    calc_len(str(c.get("tgt", c.get("trans", "")) or ""))
                    * TARGET_SUB_MULTIPLIER
                    <= MAX_SUB_LENGTH
                    for c in all_cues
                )
            ):
                report_progress(
                    1.0,
                    "复用已有字幕切分结果",
                    "字幕显示切分已是最新，跳过重复切分",
                    tag="llm",
                    label="CACHE",
                )
                return
        except (ValueError, TypeError, KeyError, OSError):
            pass

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
                "max_length": MAX_SUB_LENGTH,
                "target_multiplier": TARGET_SUB_MULTIPLIER,
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
    if len(utterances) == len(orig_src) and words:
        n_total_w = len(words)
        for u_idx, u in enumerate(utterances):
            span = u.get("word_span") or [0, n_total_w]
            w_a = max(0, min(int(span[0]) if len(span) >= 2 else 0, n_total_w - 1))
            w_b = max(
                w_a + 1,
                min(int(span[1]) if len(span) >= 2 else n_total_w, n_total_w),
            )
            if w_b - w_a > 1:
                raw_gaps = [
                    max(
                        0.0,
                        float(words[k + 1].get("start", 0.0))
                        - float(words[k].get("end", 0.0)),
                    )
                    for k in range(w_a, w_b - 1)
                ]
                src_ws = orig_src[u_idx].split()
                if (
                    " " in orig_src[u_idx]
                    and len(src_ws) >= 2
                    and len(src_ws) != (w_b - w_a)
                ):
                    end_map = _align_part_end_words(w_a, w_b, src_ws, words)
                    orig_word_gaps[u_idx] = [
                        raw_gaps[
                            min(
                                len(raw_gaps) - 1,
                                max(0, end_map.get(sw_i, w_a + sw_i) - w_a),
                            )
                        ]
                        for sw_i in range(len(src_ws) - 1)
                    ]
                else:
                    orig_word_gaps[u_idx] = raw_gaps
            else:
                orig_word_gaps[u_idx] = None

    for attempt in range(3):
        report_progress(
            0.15 + attempt * 0.25,
            f"字幕切分第 {attempt + 1}/3 轮",
            f"执行字幕切分第 {attempt + 1} 轮",
            tag="llm",
            label="SPLIT",
        )
        prev_src = src.copy()
        line_gaps: list[list[float] | None] | None = None
        if attempt == 0 and orig_word_gaps:
            line_gaps = [orig_word_gaps.get(i) for i in range(len(src))]
        split_src, split_trans, remerged = split_align_subs(
            src.copy(), trans, line_word_gaps=line_gaps
        )
        if attempt == 0:
            orig_remerged = remerged

        if len(split_src) != len(prev_src):
            new_indices: list[int] = []
            cursor = 0
            for orig_idx, old_s in zip(origin_indices, prev_src):
                if cursor < len(split_src) and split_src[cursor] == old_s:
                    new_indices.append(orig_idx)
                    cursor += 1
                else:
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

    if len(origin_indices) != len(split_src):
        origin_indices = [min(i, len(orig_src) - 1) for i in range(len(split_src))]

    grouped_src: dict[int, list[str]] = {i: [] for i in range(len(orig_src))}
    grouped_tr: dict[int, list[str]] = {i: [] for i in range(len(orig_src))}
    for u_idx, s_part, t_part in zip(origin_indices, split_src, split_trans):
        grouped_src[u_idx].append(str(s_part))
        grouped_tr[u_idx].append(str(t_part))

    for u_idx, u in enumerate(utterances):
        u["tgt"] = str(orig_remerged[u_idx])
        if not str(u.get("tts_text") or "").strip():
            u["tts_text"] = str(orig_remerged[u_idx])
        u["sub_split_config"] = {
            "max_length": MAX_SUB_LENGTH,
            "target_multiplier": TARGET_SUB_MULTIPLIER,
        }
        s_parts = grouped_src.get(u_idx) or [str(orig_src[u_idx])]
        t_parts = grouped_tr.get(u_idx) or [str(orig_remerged[u_idx])]
        if words:
            span = u.get("word_span") or [0, len(words)]
            w_a = int(span[0]) if len(span) >= 2 else 0
            w_b = int(span[1]) if len(span) >= 2 else len(words)
            u["sub_cues"] = _partition_word_span_for_cues(
                w_a, w_b, s_parts, t_parts, words
            )
        else:
            u["sub_cues"] = [
                {
                    "src": sp,
                    "tgt": tp,
                    "trans": tp,
                    "word_span": [0, 0],
                    "start": float(u.get("start") or 0.0),
                    "end": float(u.get("end") or 0.0),
                }
                for sp, tp in zip(s_parts, t_parts)
            ]

    rev = ws.save_utterances(utterances)
    ws.set_kv("sub_split_rev", rev)
    report_progress(
        1.0,
        f"已生成 {len(split_src)} 条字幕行",
        f"字幕切分完成（{len(orig_src)} 句 -> {len(split_src)} 条字幕行）",
        tag="llm",
        label="SPLIT",
    )


if __name__ == "__main__":
    split_for_sub_main()

from __future__ import annotations

import math
import re
from typing import List, Tuple

from core.subtitle_split.metrics import (
    _CJK_CHAR_RE,
    _CLAUSE_SPLIT_RE,
    _LEADING_PUNCT_RE,
    _normalize_book_marks,
    calc_len,
)
from core.subtitle_split.ratio_align import align_source_by_ratio

_CJK_SYNTACTIC_SPLIT_RE = re.compile(
    r"(?:(?<=适用于)|(?<=取决于)|(?<=相当于)|(?<=得益于)|(?<=来自于)|(?<=是由于)|(?<=是因为)|(?<=指的是)|(?<=意味着)|(?<=的时候)|(?<=的情况下)|(?<=的同时)|(?<=之后)|(?<=之前)|(?<=的)|(?=而且)|(?=并且)|(?=同时)|(?=以及)|(?=或者)|(?=从而)|(?=因此)|(?=所以)|(?=因为)|(?=如果)|(?=所[\u4e00-\u9fff]{1,2}(?:的)?))"
)


def _build_clause_dp_table(
    clauses: list[str], max_k: int, clause_lens: list[float] | None = None
) -> tuple[list[list[float]], list[list[int]]]:
    n = len(clauses)
    max_k = min(max_k, n)
    lens = clause_lens if clause_lens is not None else [calc_len(c) for c in clauses]
    weights = [max(1.0, w) for w in lens]
    prefix = [0.0]
    for w in weights:
        prefix.append(prefix[-1] + w)

    dp = [[float("inf")] * (max_k + 1) for _ in range(n + 1)]
    parent = [[0] * (max_k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp_i = dp[i]
        parent_i = parent[i]
        p_i = prefix[i]
        for j in range(1, min(i, max_k) + 1):
            best_cost = float("inf")
            best_prev = 0
            for prev in range(j - 1, i):
                seg_w = p_i - prefix[prev]
                prev_cost = dp[prev][j - 1]
                cost = prev_cost if prev_cost >= seg_w else seg_w
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev
            dp_i[j] = best_cost
            parent_i[j] = best_prev
    return dp, parent


def _reconstruct_clause_groups(
    clauses: list[str], parent: list[list[int]], k: int
) -> list[str]:
    curr = len(clauses)
    groups: list[str] = []
    for j in range(k, 0, -1):
        prev = parent[curr][j]
        groups.append("".join(clauses[prev:curr]).strip())
        curr = prev
    groups.reverse()
    return [g for g in groups if g]


def _partition_clauses_dp(clauses: list[str], k: int) -> list[str]:
    """Partition ``clauses`` into ``k`` contiguous groups minimizing the maximum group weight."""
    n = len(clauses)
    if k <= 1 or n <= 1:
        return ["".join(clauses).strip()]
    k = min(k, n)
    _, parent = _build_clause_dp_table(clauses, k)
    return _reconstruct_clause_groups(clauses, parent, k)


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


def _subsplit_overlong_cjk_clause(
    clause: str, max_sub_length: float, target_sub_multiplier: float
) -> List[str]:
    if len(clause) <= 25 and calc_len(clause) * target_sub_multiplier <= max_sub_length:
        return [clause]
    cjk_count = len(_CJK_CHAR_RE.findall(clause))
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
        left_len = calc_len(left)
        right_len = calc_len(right)
        if (
            left_len >= 12.25
            and right_len >= 12.25
            and left_len * target_sub_multiplier <= max_single_w
            and right_len * target_sub_multiplier <= max_single_w
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
    """Split ``tr_line`` and ``src_line`` locally using internal clause punctuation when
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
    n_clauses = len(clauses)
    if n_clauses < 2:
        return None
    clause_lens = [calc_len(c) for c in clauses]
    if any(cl * target_sub_multiplier > max_sub_length for cl in clause_lens):
        return None

    tr_w = calc_len(tr_clean) * target_sub_multiplier
    comfortable_target_w = min(max_sub_length, max(38.0, max_sub_length * 0.60))
    needed_k = max(
        math.ceil(tr_w / max(1.0, max_sub_length)),
        math.ceil(tr_w / comfortable_target_w),
        2,
    )
    needed_k = min(needed_k, n_clauses)

    dp, parent = _build_clause_dp_table(clauses, n_clauses, clause_lens=clause_lens)
    for k_try in range(needed_k, n_clauses + 1):
        if dp[n_clauses][k_try] * target_sub_multiplier > max_sub_length + 1e-6:
            continue
        cand_groups = _reconstruct_clause_groups(clauses, parent, k_try)
        if len(cand_groups) < 2:
            continue
        group_lens = [calc_len(g) for g in cand_groups]
        if any(
            gl < 7.5 or gl * target_sub_multiplier > max_sub_length for gl in group_lens
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

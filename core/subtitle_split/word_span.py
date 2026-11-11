from __future__ import annotations

import difflib
from typing import Any

from core.subtitle_split.metrics import calc_len, needs_subtitle_split


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


def compute_utterance_word_gaps(
    utterance: dict[str, Any],
    orig_src_line: str,
    words: list[dict[str, Any]],
) -> list[float] | None:
    if not words:
        return None
    n_total_w = len(words)
    span = utterance.get("word_span") or [0, n_total_w]
    w_a = max(0, min(int(span[0]) if len(span) >= 2 else 0, n_total_w - 1))
    w_b = max(
        w_a + 1,
        min(int(span[1]) if len(span) >= 2 else n_total_w, n_total_w),
    )
    if w_b - w_a <= 1:
        return None
    raw_gaps = [
        max(
            0.0,
            float(words[k + 1].get("start", 0.0)) - float(words[k].get("end", 0.0)),
        )
        for k in range(w_a, w_b - 1)
    ]
    src_ws = orig_src_line.split()
    if " " in orig_src_line and len(src_ws) >= 2 and len(src_ws) != (w_b - w_a):
        end_map = _align_part_end_words(w_a, w_b, src_ws, words)
        return [
            raw_gaps[
                min(
                    len(raw_gaps) - 1,
                    max(0, end_map.get(sw_i, w_a + sw_i) - w_a),
                )
            ]
            for sw_i in range(len(src_ws) - 1)
        ]
    return raw_gaps


def _slice_gaps_for_parts(
    orig_src_line: str,
    orig_gaps: list[float] | None,
    part_lines: list[str],
) -> list[list[float] | None]:
    if orig_gaps is None:
        return [None] * len(part_lines)
    if len(part_lines) == 1 and part_lines[0] == orig_src_line:
        return [orig_gaps]
    if " " in orig_src_line:
        orig_tokens = orig_src_line.split()
        part_tokens = [p.split() for p in part_lines]
    else:
        orig_tokens = list(orig_src_line.strip())
        part_tokens = [list(p.strip()) for p in part_lines]

    if (
        len(orig_gaps) != len(orig_tokens) - 1
        or [tok for pt in part_tokens for tok in pt] != orig_tokens
    ):
        return [None] * len(part_lines)

    sliced: list[list[float] | None] = []
    offset = 0
    for pt in part_tokens:
        m = len(pt)
        sliced.append(orig_gaps[offset : offset + m - 1] if m > 1 else None)
        offset += m
    return sliced


def build_attempt_line_gaps(
    utterances: list[dict[str, Any]],
    orig_src: list[str],
    words: list[dict[str, Any]],
    src_lines: list[str],
    tr_lines: list[str],
    origin_indices: list[int],
    max_sub_length: float,
    target_sub_multiplier: float,
    gap_cache: dict[int, list[float] | None],
) -> list[list[float] | None] | None:
    if (
        not words
        or len(utterances) != len(orig_src)
        or len(origin_indices) != len(src_lines)
    ):
        return None

    needed_origins: set[int] = set()
    grouped_positions: dict[int, list[int]] = {}
    for line_pos, (u_idx, s_line, t_line) in enumerate(
        zip(origin_indices, src_lines, tr_lines)
    ):
        grouped_positions.setdefault(u_idx, []).append(line_pos)
        if needs_subtitle_split(s_line, t_line, max_sub_length, target_sub_multiplier)[
            0
        ]:
            needed_origins.add(u_idx)

    line_gaps: list[list[float] | None] = [None] * len(src_lines)
    for u_idx in needed_origins:
        if u_idx not in gap_cache:
            gap_cache[u_idx] = compute_utterance_word_gaps(
                utterances[u_idx], orig_src[u_idx], words
            )
        u_gaps = gap_cache[u_idx]
        if u_gaps is None:
            continue
        positions = grouped_positions[u_idx]
        part_lines = [src_lines[pos] for pos in positions]
        for pos, sub_gaps in zip(
            positions, _slice_gaps_for_parts(orig_src[u_idx], u_gaps, part_lines)
        ):
            line_gaps[pos] = sub_gaps
    return line_gaps


def remap_origin_indices(
    origin_indices: list[int], prev_src: list[str], split_src: list[str]
) -> list[int]:
    if len(split_src) == len(prev_src):
        return origin_indices
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
    return new_indices if len(new_indices) == len(split_src) else origin_indices


def is_sub_split_cache_valid(
    ws: Any,
    utterances: list[dict[str, Any]],
    max_sub_length: float,
    target_sub_multiplier: float,
) -> bool:
    if not utterances or not ws.has_sub_cues():
        return False
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
        config_matches = prev_cfg == (max_sub_length, target_sub_multiplier)
        all_cues = [
            c
            for u in utterances
            for c in (u.get("sub_cues") or [])
            if isinstance(c, dict)
        ]
        return bool(
            rev_matches
            and config_matches
            and all_cues
            and all(
                calc_len(str(c.get("src") or "")) <= max_sub_length for c in all_cues
            )
            and all(
                calc_len(str(c.get("tgt", c.get("trans", "")) or ""))
                * target_sub_multiplier
                <= max_sub_length
                for c in all_cues
            )
        )
    except (ValueError, TypeError, KeyError, OSError):
        return False


def populate_utterance_sub_cues(
    utterances: list[dict[str, Any]],
    orig_src: list[str],
    orig_remerged: list[str],
    origin_indices: list[int],
    split_src: list[str],
    split_trans: list[str],
    words: list[dict[str, Any]],
    max_sub_length: float,
    target_sub_multiplier: float,
) -> None:
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
            "max_length": max_sub_length,
            "target_multiplier": target_sub_multiplier,
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

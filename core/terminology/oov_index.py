from __future__ import annotations

import math
import threading
from typing import Any

from .lexical import (
    _KB_STOP_TOKENS,
    _char_mask_and_dups,
    _char_sim,
    _is_standard_english,
    _phonetic_dt_collapsed,
    _phonetic_skeleton,
    _stem_word,
)


def _token_search_meta(
    w: str, cache: dict[str, tuple[str, str, str, int, int, bool]]
) -> tuple[str, str, str, int, int, bool]:
    meta = cache.get(w)
    if meta is None:
        st = _stem_word(w)
        pw = _phonetic_skeleton(w)
        dw = _phonetic_dt_collapsed(pw)
        mask, dups = _char_mask_and_dups(w)
        oov = not _is_standard_english(w)
        meta = (st, pw, dw, mask, dups, oov)
        cache[w] = meta
    return meta


def _build_oov_search_index(
    src_records: list[tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str]],
    stem_df: dict[str, int],
) -> dict[str, Any]:
    meta_cache: dict[str, tuple[str, str, str, int, int, bool]] = {}
    enriched: list[tuple[Any, ...]] = []
    inv_stem: dict[str, list[int]] = {}
    inv_c2: dict[tuple[str, int], list[int]] = {}
    inv_p2: dict[tuple[str, int], list[int]] = {}
    inv_d2: dict[tuple[str, int], list[int]] = {}

    for rec_id, (entry, e_bw, e_st, e_join) in enumerate(src_records):
        j_st, j_pw, j_dw, j_mask, j_dups, j_oov = _token_search_meta(e_join, meta_cache)
        is_single_std = len(e_bw) == 1 and _is_standard_english(entry["src"].lower())

        stems_for_rec: set[str] = {e_st[0] if len(e_bw) == 1 else j_st}
        c2_for_rec: set[tuple[str, int]] = {
            (e_join[i : i + 2], len(e_join)) for i in range(len(e_join) - 1)
        }
        p2_for_rec: set[tuple[str, int]] = {
            (j_pw[i : i + 2], len(j_pw)) for i in range(len(j_pw) - 1)
        }
        d2_for_rec: set[tuple[str, int]] = {
            (j_dw[i : i + 2], len(j_dw)) for i in range(len(j_dw) - 1)
        }

        qual_toks: list[tuple[str, str, str, str, int, int, bool]] = []
        if len(e_bw) >= 2:
            for tw, ts in zip(e_bw, e_st):
                if (
                    len(tw) >= 5
                    and ts not in _KB_STOP_TOKENS
                    and not _is_standard_english(tw)
                    and stem_df.get(ts, 99) <= 3
                ):
                    t_st, t_pw, t_dw, t_mask, t_dups, t_oov = _token_search_meta(
                        tw, meta_cache
                    )
                    qual_toks.append((tw, t_st, t_pw, t_dw, t_mask, t_dups, t_oov))
                    stems_for_rec.add(t_st)
                    for i in range(len(tw) - 1):
                        c2_for_rec.add((tw[i : i + 2], len(tw)))
                    for i in range(len(t_pw) - 1):
                        p2_for_rec.add((t_pw[i : i + 2], len(t_pw)))
                    for i in range(len(t_dw) - 1):
                        d2_for_rec.add((t_dw[i : i + 2], len(t_dw)))

        enriched.append(
            (
                entry,
                e_bw,
                e_st,
                e_join,
                j_st,
                j_pw,
                j_dw,
                j_mask,
                j_dups,
                j_oov,
                is_single_std,
                tuple(qual_toks),
            )
        )
        for sk in stems_for_rec:
            inv_stem.setdefault(sk, []).append(rec_id)
        for ck in c2_for_rec:
            inv_c2.setdefault(ck, []).append(rec_id)
        for pk in p2_for_rec:
            inv_p2.setdefault(pk, []).append(rec_id)
        for dk in d2_for_rec:
            inv_d2.setdefault(dk, []).append(rec_id)

    return {
        "enriched_records": tuple(enriched),
        "inv_stem": {k: tuple(v) for k, v in inv_stem.items()},
        "inv_c2": {k: tuple(v) for k, v in inv_c2.items()},
        "inv_p2": {k: tuple(v) for k, v in inv_p2.items()},
        "inv_d2": {k: tuple(v) for k, v in inv_d2.items()},
        "query_cache": {},
        "cache_lock": threading.Lock(),
    }


def _build_possessive_prefix_map(
    entries_map: dict[str, dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    by_head: dict[str, list[dict[str, Any]]] = {}
    for ek, e_entry in entries_map.items():
        if "'s " in ek and "(" not in ek:
            head, _ = ek.split("'s ", 1)
            if head and " " not in head:
                by_head.setdefault(head, []).append(e_entry)
    return {k: tuple(v) for k, v in by_head.items()}


def _fast_word_sim_ge(
    a: str,
    st_a: str,
    pa: str,
    da: str,
    mask_a: int,
    dups_a: int,
    a_oov: bool,
    b: str,
    st_b: str,
    pb: str,
    db: str,
    mask_b: int,
    dups_b: int,
    b_oov: bool,
    min_sim: float,
) -> float:
    if a == b or st_a == st_b:
        return 1.0
    la, lb = len(a), len(b)
    mx = la if la >= lb else lb
    mn = lb if la >= lb else la
    c_len_ub = mn / mx

    lpa, lpb = len(pa), len(pb)
    same_p0 = lpa >= 2 and lpb >= 2 and pa[0] == pb[0]
    if lpa >= 2 and lpb >= 2:
        p_len_ub = (lpa if lpa < lpb else lpb) / (lpa if lpa >= lpb else lpb)
        if same_p0 and a_oov and b_oov and lpa >= 3 and lpb >= 3:
            lda, ldb = len(da), len(db)
            d_len_ub = (
                (lda if lda < ldb else ldb) / (lda if lda >= ldb else ldb)
            ) * 0.96
            if d_len_ub > p_len_ub:
                p_len_ub = d_len_ub
        pw_ub = p_len_ub * (0.94 if same_p0 else 0.80)
    else:
        p_len_ub = 0.0
        pw_ub = 0.0

    comb_len_ub = 0.55 * c_len_ub + 0.45 * p_len_ub
    best_len_ub = c_len_ub if c_len_ub >= pw_ub else pw_ub
    if comb_len_ub > best_len_ub:
        best_len_ub = comb_len_ub
    if best_len_ub < min_sim:
        return 0.0

    max_common = (mask_a & mask_b).bit_count() + (dups_a if dups_a < dups_b else dups_b)
    c_bag_ub = (max_common / mx) if max_common < mn else c_len_ub
    comb_bag_ub = 0.55 * c_bag_ub + 0.45 * p_len_ub
    best_bag_ub = c_bag_ub if c_bag_ub >= pw_ub else pw_ub
    if comb_bag_ub > best_bag_ub:
        best_bag_ub = comb_bag_ub
    if best_bag_ub < min_sim:
        return 0.0

    if lpa >= 2 and lpb >= 2:
        p_sim = _char_sim(pa, pb)
        if same_p0 and a_oov and b_oov and lpa >= 3 and lpb >= 3:
            d_sim = _char_sim(da, db) * 0.96
            if d_sim > p_sim:
                p_sim = d_sim
        p_weighted = p_sim * (0.94 if same_p0 else 0.80)
    else:
        p_sim = 0.0
        p_weighted = 0.0

    if (
        p_weighted < min_sim
        and c_bag_ub < min_sim
        and (0.55 * c_bag_ub + 0.45 * p_sim) < min_sim
    ):
        return 0.0

    c_sim = _char_sim(a, b)
    res = c_sim if c_sim >= p_weighted else p_weighted
    comb = 0.55 * c_sim + 0.45 * p_sim
    return comb if comb > res else res


def _top2_scored_entries_for_token(
    bw: str,
    q_st: str,
    is_oov: bool,
    oov_index: dict[str, Any],
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, bool], list[dict[str, Any]]] = oov_index["query_cache"]
    cache_lock: threading.Lock = oov_index["cache_lock"]
    cache_key = (bw, is_oov)
    with cache_lock:
        cached = cache.get(cache_key)
    if cached is not None:
        return cached

    q_pw = _phonetic_skeleton(bw)
    q_dw = _phonetic_dt_collapsed(q_pw)
    q_mask, q_dups = _char_mask_and_dups(bw)
    la = len(bw)
    lpa = len(q_pw)
    lda = len(q_dw)

    inv_stem = oov_index["inv_stem"]
    inv_c2 = oov_index["inv_c2"]
    inv_p2 = oov_index["inv_p2"]
    inv_d2 = oov_index["inv_d2"]
    enriched_records = oov_index["enriched_records"]

    cand_ids: set[int] = set(inv_stem.get(q_st, ()))
    lb_min = math.ceil(0.62 * la)
    lb_max = int(la / 0.62)
    for cg in {bw[k : k + 2] for k in range(la - 1)}:
        for lb in range(lb_min, lb_max + 1):
            bucket = inv_c2.get((cg, lb))
            if bucket:
                cand_ids.update(bucket)

    if lpa >= 2:
        lpb_min = max(2, math.ceil(0.795 * lpa))
        lpb_max = int(lpa / 0.795)
        for pg in {q_pw[k : k + 2] for k in range(lpa - 1)}:
            for lpb in range(lpb_min, lpb_max + 1):
                bucket = inv_p2.get((pg, lpb))
                if bucket:
                    cand_ids.update(bucket)
        if is_oov and lpa >= 3 and lda >= 2:
            ldb_min = max(2, math.ceil(0.828125 * lda))
            ldb_max = int(lda / 0.828125)
            for dg in {q_dw[k : k + 2] for k in range(lda - 1)}:
                for ldb in range(ldb_min, ldb_max + 1):
                    bucket = inv_d2.get((dg, ldb))
                    if bucket:
                        cand_ids.update(bucket)

    tok_min_sim = 0.75 / 0.94
    scored_by_src: dict[str, tuple[float, dict[str, Any]]] = {}
    for rec_id in sorted(cand_ids):
        (
            entry,
            e_bw,
            e_st,
            e_join,
            j_st,
            j_pw,
            j_dw,
            j_mask,
            j_dups,
            j_oov,
            is_single_std,
            qual_toks,
        ) = enriched_records[rec_id]
        if not is_oov and is_single_std:
            continue
        if len(e_bw) == 1 and (e_bw[0] == bw or e_st[0] == q_st):
            sc = 1.0
        else:
            sc = _fast_word_sim_ge(
                bw,
                q_st,
                q_pw,
                q_dw,
                q_mask,
                q_dups,
                is_oov,
                e_join,
                j_st,
                j_pw,
                j_dw,
                j_mask,
                j_dups,
                j_oov,
                0.75,
            )
            for tw, t_st, t_pw, t_dw, t_mask, t_dups, t_oov in qual_toks:
                tsc = (
                    _fast_word_sim_ge(
                        bw,
                        q_st,
                        q_pw,
                        q_dw,
                        q_mask,
                        q_dups,
                        is_oov,
                        tw,
                        t_st,
                        t_pw,
                        t_dw,
                        t_mask,
                        t_dups,
                        t_oov,
                        tok_min_sim,
                    )
                    * 0.94
                )
                if tsc > sc:
                    sc = tsc
        if sc >= 0.75:
            sl = entry["src"].lower()
            if "(" in sl:
                sc *= 0.94
            prev = scored_by_src.get(sl)
            if prev is None or sc > prev[0]:
                scored_by_src[sl] = (sc, entry)

    scored = sorted(scored_by_src.values(), key=lambda x: x[0], reverse=True)
    top2 = [entry for _, entry in scored[:2]]
    with cache_lock:
        if len(cache) < 8192:
            cache[cache_key] = top2
    return top2

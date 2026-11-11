from __future__ import annotations

from typing import Any

from .lexical import (
    _KB_STOP_TOKENS,
    _WORD_RE,
    _char_sim,
    _fold_ascii,
    _is_standard_english,
    _phonetic_skeleton,
    _stem_word,
    _strip_word_punct,
    _tokenize_surface,
    _word_sim,
)
from .matcher import (
    _covered_tokens_from_hits,
    _ensure_kb_matcher,
    _match_terms_in_text,
)
from .ngram_index import UnifiedNgramKBIndex
from .oov_index import _top2_scored_entries_for_token


def _retrieve_rare_token_candidates(
    text: str,
    kb_data: dict[str, Any],
    exclude_srcs: set[str] | None = None,
    covered_tokens: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return []
    matcher = _ensure_kb_matcher(kb_data)
    ngram_idx: UnifiedNgramKBIndex | None = kb_data.get("ngram_index")
    orig_words = _WORD_RE.findall(_fold_ascii(text))
    if not orig_words:
        return []

    base_words = [_strip_word_punct(w.lower())[0] for w in orig_words]
    stems = [_stem_word(bw) if bw else "" for bw in base_words]
    n = len(base_words)

    if covered_tokens is None:
        covered_tokens = _covered_tokens_from_hits(_match_terms_in_text(text, matcher))

    candidates_by_src: dict[str, dict[str, Any]] = {}
    aligned_oov_tokens: dict[str, dict[str, Any]] = {}

    excluded_lower = (
        {str(s).strip().lower() for s in exclude_srcs if str(s).strip()}
        if exclude_srcs
        else None
    )

    def _add_cand(entry: dict[str, Any], alias_span: str) -> None:
        sl = entry["src"].lower()
        if excluded_lower is not None and sl in excluded_lower:
            return
        clean_al = alias_span.strip()
        if not clean_al or clean_al.lower() == sl:
            return
        item = candidates_by_src.get(sl)
        if item is None:
            als = list(entry.get("aliases") or [])
            if clean_al.lower() not in {a.lower() for a in als}:
                als.append(clean_al)
            candidates_by_src[sl] = {
                "src": entry["src"],
                "tgt": entry["tgt"],
                "note": str(entry.get("note") or ""),
                "aliases": als,
                "matched_spans": [clean_al],
                "fix_source": bool(entry.get("fix_source", False)),
                "source": str(entry.get("source") or "domain_kb"),
            }
        else:
            if clean_al.lower() not in {a.lower() for a in item["aliases"]}:
                item["aliases"].append(clean_al)
            if clean_al.lower() not in {a.lower() for a in item["matched_spans"]}:
                item["matched_spans"].append(clean_al)

    anchor_inv = matcher.get("anchor_inv", {})
    stem_df = matcher.get("stem_df", {})
    for i, st in enumerate(stems):
        bucket = anchor_inv.get(st)
        if not bucket:
            continue
        is_rare_anchor = stem_df.get(st, 99) <= 5
        for entry, e_bw, e_st, _ in bucket:
            m = len(e_bw)
            for pos, tok_st in enumerate(e_st):
                if tok_st != st:
                    continue
                start = i - pos
                end = start + m
                if start < 0 or end > n:
                    continue
                win_bw = base_words[start:end]
                win_st = stems[start:end]
                if tuple(win_bw) == e_bw or tuple(win_st) == e_st:
                    continue
                tok_sims: list[float] = []
                ok = True
                for qw, qs, tw, ts in zip(win_bw, win_st, e_bw, e_st):
                    if qw == tw or qs == ts:
                        tok_sims.append(1.0)
                    elif ts in _KB_STOP_TOKENS and qw in _KB_STOP_TOKENS:
                        tok_sims.append(0.85)
                    else:
                        both_std = _is_standard_english(qw) and _is_standard_english(tw)
                        if both_std:
                            pq, pt = _phonetic_skeleton(qw), _phonetic_skeleton(tw)
                            if not pq or not pt or pq[0] != pt[0]:
                                ok = False
                                break
                            sim = _word_sim(qw, tw)
                            if sim < (0.56 if is_rare_anchor else 0.78):
                                ok = False
                                break
                        else:
                            sim = _word_sim(qw, tw)
                            if sim < 0.56:
                                ok = False
                                break
                        tok_sims.append(sim)
                if ok and (sum(tok_sims) / m) >= 0.78:
                    _add_cand(entry, " ".join(orig_words[start:end]))
                    for qw, tw in zip(win_bw, e_bw):
                        if (
                            qw != tw
                            and len(qw) >= 5
                            and not _is_standard_english(qw)
                            and not _is_standard_english(tw)
                        ):
                            aligned_oov_tokens[qw] = entry

    rare_by_p0_len = matcher.get("rare_by_p0_len")
    rare_by_p0 = matcher.get("rare_by_p0", {})
    for i in range(n - 1):
        w1, w2 = base_words[i], base_words[i + 1]
        if (
            len(w1) < 3
            or len(w2) < 3
            or w1 in _KB_STOP_TOKENS
            or w2 in _KB_STOP_TOKENS
            or (w1 in covered_tokens and w2 in covered_tokens)
        ):
            continue
        joined = w1 + w2
        lj = len(joined)
        if lj < 6 or _is_standard_english(joined):
            continue
        p_join = _phonetic_skeleton(joined)
        if len(p_join) < 3:
            continue
        rt_candidates = (
            (
                item
                for rl in range(lj - 2, lj + 3)
                for item in rare_by_p0_len.get((p_join[0], rl), ())
            )
            if rare_by_p0_len is not None
            else (
                item
                for item in rare_by_p0.get(p_join[0], ())
                if abs(lj - len(item[0])) <= 2
            )
        )
        for rt, entries_list in rt_candidates:
            c_sim = _char_sim(joined, rt)
            if c_sim < 0.65:
                continue
            p_rt = _phonetic_skeleton(rt)
            min_c = 0.65 if (p_join == p_rt and len(p_join) >= 4) else 0.72
            if c_sim >= min_c and _word_sim(joined, rt) >= 0.85:
                span2 = " ".join(orig_words[i : i + 2])
                for entry in entries_list:
                    _add_cand(entry, span2)
                    _, e_st, _ = _tokenize_surface(entry["src"])
                    if i + 2 < n and stems[i + 2] in e_st:
                        _add_cand(entry, " ".join(orig_words[i : i + 3]))

    stem_ngram_map = matcher.get("stem_ngram_map", {})
    if ngram_idx is not None:
        for i in range(n - 1):
            w1, w2 = base_words[i], base_words[i + 1]
            if (
                w1 in _KB_STOP_TOKENS
                or w2 in _KB_STOP_TOKENS
                or (w1 in covered_tokens and w2 in covered_tokens)
                or (stems[i], stems[i + 1]) in stem_ngram_map
                or not (orig_words[i][0].isupper() and orig_words[i + 1][0].isupper())
            ):
                continue
            span2 = f"{orig_words[i]} {orig_words[i + 1]}"
            for _, entry in ngram_idx.query(span2, top_k=1, min_score=0.47):
                e_bw, _, _ = _tokenize_surface(entry["src"])
                if len(e_bw) < 2 or not all(
                    max(_word_sim(qw, tw) for tw in e_bw) >= 0.55 for qw in (w1, w2)
                ):
                    continue
                if len(e_bw) == 2 and any(
                    qw != tw and (qw in covered_tokens or _word_sim(qw, tw) < 0.62)
                    for qw, tw in zip((w1, w2), e_bw)
                ):
                    continue
                _add_cand(entry, span2)

    rare_inv = matcher.get("rare_token_inv", {})
    oov_index = matcher["oov_index"]
    seen_oov: set[str] = set()
    for i, bw in enumerate(base_words):
        if (
            len(bw) < 4
            or bw in _KB_STOP_TOKENS
            or bw.isdigit()
            or bw in covered_tokens
            or bw in seen_oov
        ):
            continue
        is_oov = not _is_standard_english(bw)
        is_cap_noun = (
            orig_words[i][0].isupper()
            and len(bw) >= 7
            and not bw.endswith(("est", "ing", "ed", "ly", "tion", "ment"))
        )
        if not (is_oov or is_cap_noun):
            continue
        seen_oov.add(bw)
        if bw in aligned_oov_tokens:
            _add_cand(aligned_oov_tokens[bw], orig_words[i])
            continue
        st = stems[i]
        if is_oov and st in rare_inv:
            for entry in rare_inv[st]:
                _add_cand(entry, orig_words[i])
            continue

        chosen_entries = list(_top2_scored_entries_for_token(bw, st, is_oov, oov_index))
        if ngram_idx is not None:
            min_ng = 0.39 if is_oov else 0.54
            for _, entry in ngram_idx.query(orig_words[i], top_k=1, min_score=min_ng):
                if entry["src"].lower() not in {
                    e["src"].lower() for e in chosen_entries
                }:
                    chosen_entries.append(entry)
        for entry in chosen_entries:
            _add_cand(entry, orig_words[i])
            if i + 1 < n and stems[i + 1] in _tokenize_surface(entry["src"])[1]:
                _add_cand(entry, " ".join(orig_words[i : i + 2]))

    possessive_map: dict[str, tuple[dict[str, Any], ...]] = matcher.get(
        "possessive_prefix_map", {}
    )
    for sl, cand in list(candidates_by_src.items()):
        if " " not in sl and not _is_standard_english(sl):
            for e_entry in possessive_map.get(sl, ()):
                _add_cand(e_entry, cand["src"])
                for sp in cand.get("matched_spans", []):
                    _add_cand(e_entry, sp)

    return list(candidates_by_src.values())

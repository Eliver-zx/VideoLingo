from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.utils import ask_gpt, load_key_or

from .candidates import _retrieve_rare_token_candidates
from .lexical import _tokenize_surface
from .matcher import _scan_text_for_kb_terms
from .ngram_index import UnifiedNgramKBIndex
from .normalize import _normalize_aliases
from .shorthands import (
    _derive_in_video_shorthands,
    _propagate_possessive_aliases,
    resolve_ambiguous_kb_candidates_with_llm,
)


def _register_auth_entry(
    entry: dict[str, Any],
    auth_by_lower: dict[str, dict[str, Any]],
    auth_by_stem: dict[tuple[str, ...], dict[str, Any]],
    *,
    overwrite_lower: bool = False,
) -> None:
    s = entry["src"]
    sl = s.lower()
    if overwrite_lower:
        auth_by_lower[sl] = entry
    else:
        auth_by_lower.setdefault(sl, entry)
    _, st, _ = _tokenize_surface(s)
    if st:
        auth_by_stem.setdefault(tuple(st), entry)
    for al in entry.get("aliases") or []:
        al_str = str(al).strip()
        if not al_str:
            continue
        auth_by_lower.setdefault(al_str.lower(), entry)
        _, al_st, _ = _tokenize_surface(al_str)
        if al_st:
            auth_by_stem.setdefault(tuple(al_st), entry)


def merge_terminology_results(
    chunk_results: list[tuple[int, dict[str, Any]]],
    custom_terms_list: list[dict[str, Any]],
    kb_data: dict[str, Any] | None = None,
    full_text: str = "",
    run_llm_disambiguation: bool = False,
    *,
    precomputed_scanned_terms: list[dict[str, Any]] | None = None,
    precomputed_covered_tokens: set[str] | None = None,
    ask_gpt_fn: Callable[..., Any] = ask_gpt,
    load_key_or_fn: Callable[..., Any] = load_key_or,
) -> dict[str, Any]:
    ordered_chunks = sorted(chunk_results, key=lambda x: x[0])
    themes: list[str] = []
    for _, res in ordered_chunks:
        if isinstance(res, dict):
            th = str(res.get("theme") or res.get("topic") or "").strip()
            if th and th not in themes:
                themes.append(th)
    combined_theme = " ".join(themes).strip()

    auth_by_stem: dict[tuple[str, ...], dict[str, Any]] = {}
    auth_by_lower: dict[str, dict[str, Any]] = {}
    for ct in custom_terms_list:
        s = str(ct.get("src") or "").strip()
        t = str(ct.get("tgt") or "").strip()
        if not s or not t:
            continue
        entry = {
            "src": s,
            "tgt": t,
            "note": str(ct.get("note") or "").strip(),
            "aliases": _normalize_aliases(ct.get("aliases"), s),
            "fix_source": bool(ct.get("fix_source", False)),
            "source": str(ct.get("source") or "custom"),
        }
        _register_auth_entry(entry, auth_by_lower, auth_by_stem, overwrite_lower=True)

    kb_entries = (kb_data.get("entries") or {}) if kb_data else {}
    for entry in kb_entries.values():
        _register_auth_entry(entry, auth_by_lower, auth_by_stem, overwrite_lower=False)

    llm_mentioned_surfaces: set[str] = set()
    for _, res in ordered_chunks:
        if not isinstance(res, dict):
            continue
        for item in res.get("terms") or []:
            if not isinstance(item, dict):
                continue
            s = str(item.get("src") or "").strip()
            if s:
                llm_mentioned_surfaces.add(s.lower())
            for al in _normalize_aliases(item.get("aliases"), s):
                llm_mentioned_surfaces.add(al.lower())

    scanned_aliases_by_src: dict[str, list[str]] = {}
    confirmed_kb_srcs: dict[str, None] = {}
    rare_terms: list[dict[str, Any]] = []
    if kb_data and full_text:
        scanned_terms = (
            precomputed_scanned_terms
            if precomputed_scanned_terms is not None
            else _scan_text_for_kb_terms(full_text, kb_data)
        )
        rare_terms = _retrieve_rare_token_candidates(
            full_text, kb_data, covered_tokens=precomputed_covered_tokens
        )
        for scanned in scanned_terms:
            src_low = scanned["src"].lower()
            prev_als = scanned_aliases_by_src.setdefault(src_low, [])
            prev_low = {x.lower() for x in prev_als}
            auth_entry = kb_entries.get(src_low)
            for al in scanned.get("aliases") or []:
                al_str = str(al).strip()
                if not al_str:
                    continue
                if al_str.lower() not in prev_low:
                    prev_als.append(al_str)
                    prev_low.add(al_str.lower())
                if auth_entry is not None:
                    auth_by_lower.setdefault(al_str.lower(), auth_entry)
                    _, al_st, _ = _tokenize_surface(al_str)
                    if al_st:
                        auth_by_stem.setdefault(tuple(al_st), auth_entry)

        for rc in rare_terms:
            sl = rc["src"].lower()
            m_spans = list(rc.get("matched_spans") or rc.get("aliases") or [])
            if sl in llm_mentioned_surfaces or any(
                str(sp).strip().lower() in llm_mentioned_surfaces for sp in m_spans
            ):
                confirmed_kb_srcs.setdefault(sl, None)
                prev_als = scanned_aliases_by_src.setdefault(sl, [])
                prev_low = {x.lower() for x in prev_als}
                auth_entry = kb_entries.get(sl)
                for sp in m_spans:
                    sp_str = str(sp).strip()
                    if not sp_str:
                        continue
                    if sp_str.lower() not in prev_low:
                        prev_als.append(sp_str)
                        prev_low.add(sp_str.lower())
                    if auth_entry is not None:
                        auth_by_lower.setdefault(sp_str.lower(), auth_entry)
                        _, sp_st, _ = _tokenize_surface(sp_str)
                        if sp_st:
                            auth_by_stem.setdefault(tuple(sp_st), auth_entry)

    ngram_idx: UnifiedNgramKBIndex | None = (
        kb_data.get("ngram_index") if kb_data else None
    )
    if (
        run_llm_disambiguation
        and kb_data
        and ngram_idx is not None
        and len(kb_entries) >= 10
        and full_text
    ):
        text_lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
        text_lines_low = [ln.lower() for ln in text_lines]

        def _find_context_sentence(span_str: str) -> str:
            sl = span_str.lower()
            for ln, ll in zip(text_lines, text_lines_low):
                if sl in ll:
                    return ln
            toks = [w for w in _tokenize_surface(span_str)[0] if len(w) >= 3]
            if toks:
                for ln, ll in zip(text_lines, text_lines_low):
                    if all(t in ll for t in toks):
                        return ln
            return ""

        unresolved_map: dict[str, dict[str, Any]] = {}
        auto_resolved: dict[str, dict[str, Any]] = {}

        for _, res in ordered_chunks:
            if not isinstance(res, dict):
                continue
            for item in res.get("terms") or []:
                if not isinstance(item, dict):
                    continue
                s = str(item.get("src") or "").strip()
                if not s:
                    continue
                _, st, _ = _tokenize_surface(s)
                st_tup = tuple(st) if st else ()
                if s.lower() in auth_by_lower or (st_tup and st_tup in auth_by_stem):
                    continue
                als = _normalize_aliases(item.get("aliases"), s)
                if any(a.lower() in auth_by_lower for a in als):
                    continue
                cand_scores: dict[str, float] = {}
                cand_entries: dict[str, dict[str, Any]] = {}
                for q in (s, *als):
                    for sc, ent in ngram_idx.query(q, top_k=3, min_score=0.43):
                        ek = ent["src"].lower()
                        if sc > cand_scores.get(ek, 0.0):
                            cand_scores[ek] = sc
                            cand_entries[ek] = ent
                if not cand_scores:
                    continue
                ranked = sorted(cand_scores.items(), key=lambda x: -x[1])[:3]
                best_ek, best_sc = ranked[0]
                second_sc = ranked[1][1] if len(ranked) > 1 else 0.0
                if best_sc >= 0.84 and (best_sc - second_sc) >= 0.12:
                    auto_resolved[s.lower()] = cand_entries[best_ek]
                    for a in als:
                        auto_resolved[a.lower()] = cand_entries[best_ek]
                else:
                    ctx = _find_context_sentence(
                        als[0] if als else s
                    ) or _find_context_sentence(s)
                    unresolved_map[s.lower()] = {
                        "span": s,
                        "aliases": als,
                        "sentence": ctx,
                        "candidates": [
                            {
                                "src": cand_entries[ek]["src"],
                                "tgt": cand_entries[ek]["tgt"],
                                "note": cand_entries[ek].get("note", ""),
                                "score": round(sc, 3),
                            }
                            for ek, sc in ranked
                        ],
                    }

        for rc in rare_terms:
            sl = rc["src"].lower()
            m_spans = list(rc.get("matched_spans") or rc.get("aliases") or [])
            if sl in llm_mentioned_surfaces or any(
                str(sp).strip().lower() in llm_mentioned_surfaces for sp in m_spans
            ):
                continue
            for sp in m_spans:
                sp_str = str(sp).strip()
                sp_low = sp_str.lower()
                if not sp_low or sp_low in auth_by_lower or sp_low in unresolved_map:
                    continue
                cands_list = [
                    {
                        "src": rc["src"],
                        "tgt": rc["tgt"],
                        "note": rc.get("note", ""),
                        "score": 0.75,
                    }
                ]
                for sc, ent in ngram_idx.query(sp_str, top_k=3, min_score=0.45):
                    if ent["src"].lower() != sl and len(cands_list) < 3:
                        cands_list.append(
                            {
                                "src": ent["src"],
                                "tgt": ent["tgt"],
                                "note": ent.get("note", ""),
                                "score": round(sc, 3),
                            }
                        )
                unresolved_map[sp_low] = {
                    "span": sp_str,
                    "aliases": [],
                    "sentence": _find_context_sentence(sp_str),
                    "candidates": cands_list,
                }

        for sp_low, ent in auto_resolved.items():
            ek = ent["src"].lower()
            confirmed_kb_srcs.setdefault(ek, None)
            auth_by_lower.setdefault(sp_low, ent)
            prev_als = scanned_aliases_by_src.setdefault(ek, [])
            if sp_low != ek and sp_low not in {x.lower() for x in prev_als}:
                prev_als.append(sp_low)

        if unresolved_map:
            llm_decisions = resolve_ambiguous_kb_candidates_with_llm(
                list(unresolved_map.values()),
                ask_gpt_fn=ask_gpt_fn,
                load_key_or_fn=load_key_or_fn,
            )
            for sp_low, chosen_src in llm_decisions.items():
                ent = kb_entries.get(chosen_src.lower())
                if ent is None:
                    continue
                ek = ent["src"].lower()
                confirmed_kb_srcs.setdefault(ek, None)
                auth_by_lower[sp_low] = ent
                prev_als = scanned_aliases_by_src.setdefault(ek, [])
                orig_span = unresolved_map.get(sp_low, {}).get("span", sp_low)
                if orig_span.lower() != ek and orig_span.lower() not in {
                    x.lower() for x in prev_als
                }:
                    prev_als.append(orig_span)
                for a in unresolved_map.get(sp_low, {}).get("aliases") or []:
                    al_low = a.lower()
                    auth_by_lower.setdefault(al_low, ent)
                    if al_low != ek and al_low not in {x.lower() for x in prev_als}:
                        prev_als.append(a)

    merged_by_src: dict[str, dict[str, Any]] = {}
    stem_to_src_key: dict[tuple[str, ...], str] = {}

    def _upsert_term(
        src: str,
        tgt: str,
        note: str = "",
        aliases: list[str] | None = None,
        fix_source: bool = False,
        source: str = "task",
        overwrite_tgt: bool = False,
    ) -> None:
        s, t = src.strip(), tgt.strip()
        if not s or not t:
            return
        _, st, _ = _tokenize_surface(s)
        st_tup = tuple(st) if st else ()

        auth = auth_by_lower.get(s.lower()) or (
            auth_by_stem.get(st_tup) if st_tup else None
        )
        raw_aliases = list(aliases or [])
        if auth is None:
            for a in raw_aliases:
                auth = auth_by_lower.get(str(a).strip().lower())
                if auth is not None:
                    break
        extra_aliases: list[str] = []
        if auth is not None:
            canonical_low = auth["src"].lower()
            for a in raw_aliases:
                a_str = str(a).strip()
                if not a_str:
                    continue
                a_auth = auth_by_lower.get(a_str.lower())
                if a_auth is not None and a_auth["src"].lower() != canonical_low:
                    continue
                if any(
                    len(tok) >= 5
                    and tok in auth_by_lower
                    and auth_by_lower[tok]["src"].lower() != canonical_low
                    for tok in _tokenize_surface(a_str)[0]
                ):
                    continue
                extra_aliases.append(a_str)
            if s.lower() != canonical_low:
                s_auth = auth_by_lower.get(s.lower())
                if s_auth is None or s_auth["src"].lower() == canonical_low:
                    extra_aliases.insert(0, s)
            s, t = auth["src"], auth["tgt"]
            if not note:
                note = str(auth.get("note") or "")
            fix_source = bool(auth.get("fix_source", False)) or fix_source
            auth_aliases = (
                scanned_aliases_by_src.get(s.lower(), [])
                if (kb_data and full_text and auth.get("source") != "custom")
                else (auth.get("aliases") or [])
            )
            extra_aliases = [*auth_aliases, *extra_aliases]
            _, st, _ = _tokenize_surface(s)
            st_tup = tuple(st) if st else ()
        else:
            extra_aliases = raw_aliases

        key = s.lower()
        if key not in merged_by_src and st_tup and st_tup in stem_to_src_key:
            t = merged_by_src[stem_to_src_key[st_tup]]["tgt"]

        clean_aliases = _normalize_aliases(extra_aliases, s)
        if key not in merged_by_src:
            merged_by_src[key] = {
                "src": s,
                "tgt": t,
                "note": note,
                "aliases": clean_aliases,
                "fix_source": fix_source,
                "source": source,
            }
            if st_tup:
                stem_to_src_key[st_tup] = key
        else:
            cur = merged_by_src[key]
            if overwrite_tgt:
                cur["tgt"] = t
            if note and not cur.get("note"):
                cur["note"] = note
            cur["fix_source"] = bool(cur.get("fix_source", False) or fix_source)
            cur["aliases"] = _normalize_aliases(
                [*(cur.get("aliases") or []), *clean_aliases], cur["src"]
            )
            if source == "task":
                cur["source"] = "task"

    for ct in custom_terms_list:
        if ct.get("source") == "custom":
            _upsert_term(
                ct["src"],
                ct["tgt"],
                str(ct.get("note") or ""),
                ct.get("aliases") or [],
                bool(ct.get("fix_source", False)),
                source="custom",
                overwrite_tgt=True,
            )

    for s_low in confirmed_kb_srcs:
        if s_low not in merged_by_src and s_low in kb_entries:
            ent = kb_entries[s_low]
            _upsert_term(
                ent["src"],
                ent["tgt"],
                str(ent.get("note") or ""),
                scanned_aliases_by_src.get(s_low, []),
                bool(ent.get("fix_source", False)),
                source="task",
            )

    for _, res in ordered_chunks:
        if not isinstance(res, dict):
            continue
        for item in res.get("terms") or []:
            if not isinstance(item, dict):
                continue
            s = str(item.get("src") or "").strip()
            t = str(item.get("tgt") or "").strip()
            n_val = item.get("note")
            n_str = (
                ""
                if n_val is None or str(n_val).lower() == "nan"
                else str(n_val).strip()
            )
            _upsert_term(s, t, n_str, _normalize_aliases(item.get("aliases"), s))

    _propagate_possessive_aliases(merged_by_src)
    _derive_in_video_shorthands(merged_by_src, full_text)
    return {"theme": combined_theme, "terms": list(merged_by_src.values())}

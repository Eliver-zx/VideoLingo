from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .lexical import (
    _CJK_RE,
    _EXTRA_COMMON_WORDS,
    _KB_STOP_TOKENS,
    _SIGNIFICANT_SYMBOL_RE,
    _WORD_RE,
    _fold_ascii,
    _is_standard_english,
    _phonetic_skeleton,
    _stem_word,
    _strip_word_punct,
    _tokenize_surface,
)
from .ngram_index import UnifiedNgramKBIndex
from .normalize import _normalize_aliases
from .oov_index import _build_oov_search_index, _build_possessive_prefix_map


@dataclass(frozen=True, slots=True)
class SurfaceRule:
    entry: dict[str, Any]
    surface: str
    is_canonical_src: bool
    sig: tuple[str, str]


@dataclass(slots=True)
class TermHit:
    entry: dict[str, Any]
    src_matched: bool = False
    matched_aliases: list[str] = field(default_factory=list)
    first_token_idx: int = 0


def _append_rule(
    bucket_map: dict[Any, list[SurfaceRule]],
    key: Any,
    rule: SurfaceRule,
) -> None:
    bucket = bucket_map.get(key)
    if bucket is None:
        bucket_map[key] = [rule]
        return
    sig = rule.sig
    for r in bucket:
        if r.sig == sig:
            return
    bucket.append(rule)


def _build_term_matcher(
    entries: list[dict[str, Any]],
    max_rare_token_df: int = 8,
    include_rare_index: bool = True,
) -> dict[str, Any]:
    exact_ngram_map: dict[tuple[str, ...], list[SurfaceRule]] = {}
    stem_ngram_map: dict[tuple[str, ...], list[SurfaceRule]] = {}
    first_stem_lengths: dict[str, set[int]] = {}
    compact_stem_map: dict[str, list[SurfaceRule]] = {}
    symbol_cjk_terms: list[tuple[str, re.Pattern[str] | None, SurfaceRule]] = []
    raw_token_inv: dict[str, list[dict[str, Any]]] = {}
    stem_df: dict[str, int] = {}
    src_records: list[tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str]] = []

    for entry in entries:
        src = str(entry.get("src") or "").strip()
        tgt = str(entry.get("tgt") or "").strip()
        if not src or not tgt:
            continue
        sig = (src.lower(), tgt.lower())
        aliases = _normalize_aliases(entry.get("aliases"), src)
        surface_specs = [(src, True), *((al, False) for al in aliases)]
        seen_st_for_df: set[str] = set()

        for surf_str, is_src in surface_specs:
            s_clean = surf_str.strip()
            if not s_clean:
                continue
            rule = SurfaceRule(
                entry=entry, surface=s_clean, is_canonical_src=is_src, sig=sig
            )
            if _CJK_RE.search(s_clean) or _SIGNIFICANT_SYMBOL_RE.search(s_clean):
                pat = None
                if not _CJK_RE.search(s_clean):
                    pat = re.compile(
                        rf"(?<![A-Za-z0-9]){re.escape(s_clean.lower())}(?![A-Za-z0-9])"
                    )
                symbol_cjk_terms.append((s_clean.lower(), pat, rule))
                continue

            bw, st, rw = _tokenize_surface(s_clean)
            if not st or len(st) > 8 or all(w in _KB_STOP_TOKENS for w in bw):
                continue

            _append_rule(exact_ngram_map, tuple(bw), rule)
            _append_rule(stem_ngram_map, tuple(st), rule)
            first_stem_lengths.setdefault(st[0], set()).add(len(st))

            joined_bw = "".join(bw)
            joined_rw = "".join(rw)
            joined_st = "".join(bw[:-1] + [st[-1]])
            for ck in (
                joined_bw,
                joined_rw,
                joined_st,
                _stem_word(joined_bw),
                _stem_word(joined_rw),
            ):
                if len(ck) >= 3 and ck not in _KB_STOP_TOKENS:
                    _append_rule(compact_stem_map, ck, rule)

            if include_rare_index and (is_src or len(joined_bw) >= 4):
                src_records.append((entry, tuple(bw), tuple(st), joined_bw))
                for tok in set(st):
                    if (
                        len(tok) >= 3
                        and tok not in _KB_STOP_TOKENS
                        and tok not in seen_st_for_df
                    ):
                        seen_st_for_df.add(tok)
                        stem_df[tok] = stem_df.get(tok, 0) + 1
                    if len(tok) >= 4 and tok not in _KB_STOP_TOKENS:
                        t_bucket = raw_token_inv.get(tok)
                        if t_bucket is None:
                            raw_token_inv[tok] = [entry]
                        elif len(t_bucket) <= max_rare_token_df and not any(
                            x["src"] == src for x in t_bucket
                        ):
                            t_bucket.append(entry)

    result: dict[str, Any] = {
        "exact_ngram_map": {k: tuple(v) for k, v in exact_ngram_map.items()},
        "stem_ngram_map": {k: tuple(v) for k, v in stem_ngram_map.items()},
        "first_stem_lengths": {k: frozenset(v) for k, v in first_stem_lengths.items()},
        "compact_stem_map": {k: tuple(v) for k, v in compact_stem_map.items()},
        "symbol_cjk_terms": tuple(symbol_cjk_terms),
    }
    if not include_rare_index:
        return result

    rare_token_inv = {
        tok: tuple(bucket)
        for tok, bucket in raw_token_inv.items()
        if 1 <= len(bucket) <= max_rare_token_df
    }
    anchor_inv: dict[
        str, list[tuple[dict[str, Any], tuple[str, ...], tuple[str, ...], str]]
    ] = {}
    for entry, bw_tup, st_tup, joined_bw in src_records:
        if len(bw_tup) < 2:
            continue
        content_stems = [
            s for s in set(st_tup) if len(s) >= 3 and s not in _KB_STOP_TOKENS
        ]
        if len(content_stems) < 2:
            continue
        for s in content_stems:
            if stem_df.get(s, 0) <= 25:
                anchor_inv.setdefault(s, []).append((entry, bw_tup, st_tup, joined_bw))

    rare_by_p0: dict[str, list[tuple[str, tuple[dict[str, Any], ...]]]] = {}
    rare_by_p0_len: dict[
        tuple[str, int], list[tuple[str, tuple[dict[str, Any], ...]]]
    ] = {}
    for rt, entries_tup in rare_token_inv.items():
        if len(rt) >= 6 and not _is_standard_english(rt):
            p_rt = _phonetic_skeleton(rt)
            if p_rt:
                rare_by_p0.setdefault(p_rt[0], []).append((rt, entries_tup))
                rare_by_p0_len.setdefault((p_rt[0], len(rt)), []).append(
                    (rt, entries_tup)
                )

    result.update(
        {
            "rare_token_inv": rare_token_inv,
            "stem_df": stem_df,
            "anchor_inv": {k: tuple(v) for k, v in anchor_inv.items()},
            "rare_by_p0": {k: tuple(v) for k, v in rare_by_p0.items()},
            "rare_by_p0_len": {k: tuple(v) for k, v in rare_by_p0_len.items()},
            "surface_records": tuple(src_records),
            "oov_index": _build_oov_search_index(src_records, stem_df),
        }
    )
    return result


def _match_terms_in_text(text: str, matcher: dict[str, Any]) -> list[TermHit]:
    if not text or not text.strip():
        return []

    exact_ngram_map = matcher.get("exact_ngram_map", {})
    stem_ngram_map = matcher["stem_ngram_map"]
    first_stem_lengths = matcher["first_stem_lengths"]
    compact_stem_map = matcher["compact_stem_map"]
    symbol_cjk_terms = matcher["symbol_cjk_terms"]
    hits_by_sig: dict[tuple[str, str], TermHit] = {}

    def _record_channel(
        rule: SurfaceRule,
        token_idx: int,
        via_compact_channel: bool = False,
        span_text: str = "",
    ) -> None:
        sig = rule.sig
        hit = hits_by_sig.get(sig)
        if hit is None:
            hit = TermHit(entry=rule.entry, first_token_idx=token_idx)
            hits_by_sig[sig] = hit
        elif token_idx < hit.first_token_idx:
            hit.first_token_idx = token_idx

        if not via_compact_channel and rule.is_canonical_src:
            hit.src_matched = True
        else:
            alias_str = (
                span_text.strip()
                if (via_compact_channel and rule.is_canonical_src)
                else rule.surface.strip()
            )
            if (
                alias_str
                and alias_str.lower() != rule.entry["src"].lower()
                and alias_str.lower() not in {a.lower() for a in hit.matched_aliases}
            ):
                hit.matched_aliases.append(alias_str)

    text_lower = text.lower()
    for s_low, pat, rule in symbol_cjk_terms:
        if s_low in text_lower and (pat is None or pat.search(text_lower)):
            _record_channel(rule, token_idx=0, via_compact_channel=False)

    orig_words = _WORD_RE.findall(_fold_ascii(text))
    if not orig_words:
        return sorted(hits_by_sig.values(), key=lambda h: h.first_token_idx)

    base_words: list[str] = []
    raw_words: list[str] = []
    for w in orig_words:
        bw, rw = _strip_word_punct(w.lower())
        base_words.append(bw)
        raw_words.append(rw)
    stems = [_stem_word(bw) if bw else "" for bw in base_words]
    valid_comp = [len(bw) >= 3 and bw not in _KB_STOP_TOKENS for bw in base_words]

    n = len(stems)
    covered = [False] * n
    for k in range(8, 0, -1):
        for i in range(n - k + 1):
            if covered[i]:
                continue
            lens = first_stem_lengths.get(stems[i])
            if lens and k in lens and not any(covered[i : i + k]):
                bucket = exact_ngram_map.get(
                    tuple(base_words[i : i + k])
                ) or stem_ngram_map.get(tuple(stems[i : i + k]))
                if bucket:
                    for j in range(i, i + k):
                        covered[j] = True
                    span = " ".join(orig_words[i : i + k])
                    for rule in bucket:
                        _record_channel(rule, i, False, span)
                    continue

            if k == 1 and not covered[i]:
                c_bucket = (
                    compact_stem_map.get(base_words[i])
                    or compact_stem_map.get(stems[i])
                    or compact_stem_map.get(raw_words[i])
                )
                if c_bucket:
                    covered[i] = True
                    for rule in c_bucket:
                        _record_channel(rule, i, True, orig_words[i])
            elif (
                2 <= k <= 3
                and valid_comp[i]
                and valid_comp[i + 1]
                and (k == 2 or valid_comp[i + 2])
                and not any(covered[i : i + k])
            ):
                span_bw = base_words[i : i + k]
                c_bucket = compact_stem_map.get(
                    "".join(span_bw[:-1]) + stems[i + k - 1]
                ) or compact_stem_map.get("".join(span_bw))
                if c_bucket:
                    for j in range(i, i + k):
                        covered[j] = True
                    span = " ".join(orig_words[i : i + k])
                    for rule in c_bucket:
                        _record_channel(rule, i, True, span)

    return sorted(hits_by_sig.values(), key=lambda h: h.first_token_idx)


def format_term_hits_for_prompt(hits: list[TermHit]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        src = hit.entry["src"]
        tgt = hit.entry["tgt"]
        note = str(hit.entry.get("note") or "").strip()
        note_suffix = f" [{note}]" if note else ""
        src_lines = [f"{src}:{tgt}{note_suffix}"] if hit.src_matched else []
        alias_lines: list[str] = []
        for al in hit.matched_aliases:
            al_note = note_suffix
            if (
                al.lower().endswith("less")
                and not src.lower().endswith("less")
                and "without" not in note.lower()
            ):
                al_note = f" [without {src}; {note}]" if note else f" [without {src}]"
            alias_lines.append(f"{al}:{tgt} ({src}){al_note}")
        for line in (*src_lines, *alias_lines):
            k = line.lower()
            if k not in seen:
                seen.add(k)
                lines.append(line)
    return lines


def _ensure_kb_matcher(kb_data: dict[str, Any]) -> dict[str, Any]:
    entries_map: dict[str, dict[str, Any]] = kb_data.get("entries") or {}
    matcher = kb_data.get("matcher")
    if matcher is None or "surface_records" not in matcher:
        matcher = _build_term_matcher(
            list(entries_map.values()), include_rare_index=True
        )
        kb_data["matcher"] = matcher
    elif "oov_index" not in matcher:
        matcher["oov_index"] = _build_oov_search_index(
            list(matcher["surface_records"]), matcher.get("stem_df", {})
        )
    if "possessive_prefix_map" not in matcher:
        matcher["possessive_prefix_map"] = _build_possessive_prefix_map(entries_map)
    if "ngram_index" not in kb_data and entries_map:
        kb_data["ngram_index"] = UnifiedNgramKBIndex(list(entries_map.values()))
    return matcher


def _covered_tokens_from_hits(hits: list[TermHit]) -> set[str]:
    covered: set[str] = set()
    for hit in hits:
        for surf in (hit.entry["src"], *hit.matched_aliases):
            covered.update(_tokenize_surface(surf)[0])
    return covered


def _hits_to_scanned_terms(hits: list[TermHit]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for hit in hits:
        src = hit.entry["src"]
        source = str(hit.entry.get("source") or "domain_kb")
        if source == "domain_dict":
            bw, _, _ = _tokenize_surface(src)
            content_bw = [w for w in bw if w not in _KB_STOP_TOKENS]
            if len(content_bw) <= 1 and all(
                w in _EXTRA_COMMON_WORDS for w in content_bw
            ):
                continue
        results.append(
            {
                "src": src,
                "tgt": hit.entry["tgt"],
                "note": str(hit.entry.get("note") or ""),
                "aliases": list(hit.matched_aliases),
                "src_matched": hit.src_matched,
                "fix_source": bool(hit.entry.get("fix_source", False)),
                "source": source,
            }
        )
    return results


def _scan_text_with_covered_tokens(
    text: str, kb_data: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[str]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return [], set()
    hits = _match_terms_in_text(text, _ensure_kb_matcher(kb_data))
    return _hits_to_scanned_terms(hits), _covered_tokens_from_hits(hits)


def _scan_text_for_kb_terms(text: str, kb_data: dict[str, Any]) -> list[dict[str, Any]]:
    if not text or not kb_data or not kb_data.get("entries"):
        return []
    return _hits_to_scanned_terms(
        _match_terms_in_text(text, _ensure_kb_matcher(kb_data))
    )

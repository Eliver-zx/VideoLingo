import re
from typing import Any

from core.nlp_split.projection import _tokenize_words_for_alignment
from core.nlp_split.tokenizer import (
    CLAUSE_PUNCT_CHARS,
    SENTENCE_END_CHARS,
    _TRAILING_QUOTE_CHARS,
    _clean_word,
    _is_true_sentence_end_on_word,
)
from core.utils import join_words

CLAUSE_STITCH_MAX_GAP = 0.35
CLAUSE_SPLIT_MIN_GAP = 0.25
LONG_SENTENCE_MAX_WORDS = 120
LONG_SENTENCE_MAX_CJK_CHARS = 180
LONG_SENTENCE_MAX_DURATION = 25.0
STITCH_MAX_COMBINED_WORDS = 60

_FRONTED_CLAUSE_STARTERS = set(
    "while although though even because cuz cause since unless until whereas "
    "whenever wherever whether if when as after before once".split()
)
_CLAUSE_CONJUNCTIONS = _FRONTED_CLAUSE_STARTERS | set(
    "which who whom whose where especially particularly specifically including "
    "excluding meaning".split()
)
_DISCOURSE_LEADINS = set("so and but now well ok okay hey yeah yes no".split())
_DANGLING_TAILS = _CLAUSE_CONJUNCTIONS | set(
    "that and but or nor so yet for to of in on at by with from into onto upon about above "
    "below under over between among through during without within against toward towards "
    "a an the my your his her its our their this these those is are was were be "
    "been being have has had do does did can could will would shall should may might must".split()
)
_FILLER_TOKENS = set("uh um er ah hmm uhh umm".split())

_HAN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

_TermsFingerprint = tuple[int, tuple[tuple[str, tuple[str, ...]], ...]]
_CUSTOM_ALIAS_CACHE: (
    tuple[_TermsFingerprint, list[list[str]], dict[str, list[list[str]]]] | None
) = None


def _extract_patterns_from_entries(
    entries: list[dict[str, Any]],
    seen: set[tuple[str, ...]],
    out: list[list[str]],
) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for cand in (entry.get("src"), *(entry.get("aliases") or ())):
            if not cand:
                continue
            toks, _, _ = _tokenize_words_for_alignment([str(cand)])
            if len(toks) >= 2 and (key := tuple(toks)) not in seen:
                seen.add(key)
                out.append(toks)


def _index_multiword_patterns(patterns: list[list[str]]) -> dict[str, list[list[str]]]:
    index: dict[str, list[list[str]]] = {}
    for pat in patterns:
        if len(pat) >= 2:
            index.setdefault(pat[0], []).append(pat)
    for bucket in index.values():
        bucket.sort(key=len, reverse=True)
    return index


def _load_workspace_and_custom_sources() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    from core import _3_1_split_nlp
    from core._4_1_summarize import load_custom_terms

    ws_entries: list[dict[str, Any]] = []
    ws = _3_1_split_nlp.get_active_workspace()
    if ws is not None and hasattr(ws, "has_terms") and ws.has_terms():
        ws_entries = list(ws.load_terms().get("terms") or [])
    custom_entries = load_custom_terms()
    return ws_entries, custom_entries


def _get_cached_custom_aliases(
    custom_entries: list[dict[str, Any]],
) -> tuple[list[list[str]], dict[str, list[list[str]]]]:
    global _CUSTOM_ALIAS_CACHE
    if not custom_entries:
        return [], {}
    fp: _TermsFingerprint = (
        len(custom_entries),
        tuple(
            (
                str(e.get("src") or ""),
                tuple(str(a) for a in (e.get("aliases") or ()) if a),
            )
            for e in custom_entries
            if isinstance(e, dict)
        ),
    )
    if _CUSTOM_ALIAS_CACHE is not None and _CUSTOM_ALIAS_CACHE[0] == fp:
        return _CUSTOM_ALIAS_CACHE[1], _CUSTOM_ALIAS_CACHE[2]
    seen: set[tuple[str, ...]] = set()
    patterns: list[list[str]] = []
    _extract_patterns_from_entries(custom_entries, seen, patterns)
    index = _index_multiword_patterns(patterns)
    _CUSTOM_ALIAS_CACHE = (fp, patterns, index)
    return patterns, index


def _resolve_multiword_aliases(
    extra_terms: list[dict[str, Any]] | None = None,
) -> tuple[list[list[str]], dict[str, list[list[str]]]]:
    ws_entries, custom_entries = _load_workspace_and_custom_sources()
    if extra_terms is None:
        custom_patterns, custom_index = _get_cached_custom_aliases(custom_entries)
        if not ws_entries:
            return list(custom_patterns), custom_index
        seen: set[tuple[str, ...]] = set()
        patterns: list[list[str]] = []
        _extract_patterns_from_entries(ws_entries, seen, patterns)
        for pat in custom_patterns:
            if (key := tuple(pat)) not in seen:
                seen.add(key)
                patterns.append(pat)
        return patterns, _index_multiword_patterns(patterns)

    seen = set()
    patterns = []
    for group in (extra_terms, ws_entries, custom_entries):
        _extract_patterns_from_entries(group, seen, patterns)
    return patterns, _index_multiword_patterns(patterns)


def _load_known_multiword_aliases(
    extra_terms: list[dict[str, Any]] | None = None,
) -> list[list[str]]:
    """Collect multi-word token sequences from workspace and custom/domain glossaries."""
    return _resolve_multiword_aliases(extra_terms)[0]


def _protect_multiword_alias_spans(
    cleaned_words: list[str],
    end_indices: set[int],
    punct_by_word: dict[int, str],
    extra_terms: list[dict[str, Any]] | None = None,
) -> None:
    """Prevent sentence breaks inside matched multi-word glossary terms or aliases."""
    if len(cleaned_words) < 2:
        return
    _, pattern_index = _resolve_multiword_aliases(extra_terms)
    if not pattern_index:
        return
    word_clean = [_clean_word(w) for w in cleaned_words]
    n = len(word_clean)
    for start in range(n - 1):
        candidates = pattern_index.get(word_clean[start])
        if not candidates:
            continue
        for pat in candidates:
            m = len(pat)
            if start + m <= n and all(
                word_clean[start + off] == pat[off] for off in range(1, m)
            ):
                for internal_idx in range(start, start + m - 1):
                    end_indices.discard(internal_idx)
                    p_internal = punct_by_word.get(internal_idx, "")
                    if p_internal and p_internal in SENTENCE_END_CHARS:
                        punct_by_word.pop(internal_idx, None)
                break


def _stitch_incomplete_clause_spans(
    spans: list[tuple[int, int]],
    cleaned_words: list[str],
    punct_by_word: dict[int, str],
    word_gaps: list[float] | None,
) -> list[tuple[int, int]]:
    """Merge adjacent spans when separated by a short acoustic pause and subordinate/dangling syntax."""
    if len(spans) <= 1:
        return spans

    comma_char = (
        "，" if any(_HAN_CHAR_RE.search(w) for w in cleaned_words[:20]) else ","
    )

    def _first_meaningful_word(s_idx: int, e_idx: int) -> str:
        for k in range(s_idx, min(e_idx + 1, s_idx + 4)):
            w = _clean_word(cleaned_words[k])
            if w and w not in _FILLER_TOKENS:
                return w
        return _clean_word(cleaned_words[s_idx])

    def _has_fronted_starter(s_idx: int, e_idx: int) -> bool:
        for k in range(s_idx, min(e_idx + 1, s_idx + 4)):
            w = _clean_word(cleaned_words[k])
            if not w or w in _FILLER_TOKENS or w in _DISCOURSE_LEADINS:
                continue
            return w in _FRONTED_CLAUSE_STARTERS
        return False

    def _has_clause_punct(s_idx: int, e_idx: int) -> bool:
        return any(
            (pk := punct_by_word.get(k, "")) and pk in CLAUSE_PUNCT_CHARS
            for k in range(s_idx, e_idx)
        )

    stitched: list[tuple[int, int]] = [spans[0]]
    for nxt_s, nxt_e in spans[1:]:
        cur_s, cur_e = stitched[-1]
        cur_len = cur_e - cur_s + 1
        nxt_len = nxt_e - nxt_s + 1
        combined_len = nxt_e - cur_s + 1
        gap = word_gaps[cur_e] if word_gaps and cur_e < len(word_gaps) else 0.0
        cur_last = _clean_word(cleaned_words[cur_e])
        nxt_first = _first_meaningful_word(nxt_s, nxt_e)
        end_p = punct_by_word.get(cur_e, "")

        has_internal_comma = _has_clause_punct(cur_s, cur_e)
        nxt_has_internal_comma = _has_clause_punct(nxt_s, nxt_e)
        should_merge = False
        if combined_len <= STITCH_MAX_COMBINED_WORDS and (
            not end_p or end_p not in "?!？！"
        ):
            if cur_last in _DANGLING_TAILS or cur_last in _FILLER_TOKENS:
                should_merge = gap <= 0.65
            elif (
                _has_fronted_starter(cur_s, cur_e)
                and cur_len <= 18
                and not has_internal_comma
                and gap <= CLAUSE_STITCH_MAX_GAP
            ):
                should_merge = True
            elif (
                nxt_first in _CLAUSE_CONJUNCTIONS
                and nxt_len <= 20
                and gap <= CLAUSE_STITCH_MAX_GAP
                and not (
                    nxt_first in _FRONTED_CLAUSE_STARTERS and nxt_has_internal_comma
                )
            ):
                should_merge = True

        if should_merge:
            punct_by_word[cur_e] = comma_char
            stitched[-1] = (cur_s, nxt_e)
        else:
            stitched.append((nxt_s, nxt_e))

    return stitched


def _split_long_span_at_clauses(
    start_idx: int,
    end_idx: int,
    cleaned_words: list[str],
    punct_by_word: dict[int, str],
    word_dicts: list[dict[str, Any]] | None,
    word_gaps: list[float] | None,
) -> list[tuple[int, int]]:
    """Split overly long sentences at internal clause punctuation."""
    n_span_words = end_idx - start_idx + 1
    cjk_chars = sum(
        len(_CJK_CHAR_RE.findall(cleaned_words[k]))
        for k in range(start_idx, end_idx + 1)
    )
    duration = (
        float(word_dicts[end_idx].get("end", 0.0))
        - float(word_dicts[start_idx].get("start", 0.0))
        if word_dicts and end_idx < len(word_dicts)
        else 0.0
    )
    is_overlong = (
        n_span_words > LONG_SENTENCE_MAX_WORDS
        or cjk_chars > LONG_SENTENCE_MAX_CJK_CHARS
    ) and (duration > LONG_SENTENCE_MAX_DURATION if word_dicts else True)
    if not is_overlong or n_span_words < 6:
        return [(start_idx, end_idx)]

    clause_candidates: list[int] = []
    for k in range(start_idx + 2, end_idx - 1):
        p = punct_by_word.get(k, "")
        w_tail = cleaned_words[k][-1] if cleaned_words[k] else ""
        if (p and p in CLAUSE_PUNCT_CHARS) or (w_tail and w_tail in CLAUSE_PUNCT_CHARS):
            next_tok = _clean_word(cleaned_words[k + 1]) if k + 1 <= end_idx else ""
            if next_tok not in _CLAUSE_CONJUNCTIONS:
                clause_candidates.append(k)

    if not clause_candidates:
        return [(start_idx, end_idx)]

    mid = (start_idx + end_idx) / 2.0
    pause_candidates = [
        k
        for k in clause_candidates
        if word_gaps is not None
        and k < len(word_gaps)
        and word_gaps[k] >= CLAUSE_SPLIT_MIN_GAP
    ]
    pool = pause_candidates or clause_candidates
    best_k = min(pool, key=lambda k: abs(k - mid))

    return _split_long_span_at_clauses(
        start_idx, best_k, cleaned_words, punct_by_word, word_dicts, word_gaps
    ) + _split_long_span_at_clauses(
        best_k + 1, end_idx, cleaned_words, punct_by_word, word_dicts, word_gaps
    )


def _normalize_slice_words(slice_words: list[str], end_punct: str) -> list[str]:
    """Strip spurious mid-sentence periods while preserving abbreviations and decimals."""
    out = list(slice_words)
    n = len(out)
    for i, w in enumerate(out):
        s = str(w).strip()
        if not s:
            continue
        stripped = s.rstrip(_TRAILING_QUOTE_CHARS)
        quote_tail = s[len(stripped) :]
        if i < n - 1:
            if stripped and stripped[-1] in SENTENCE_END_CHARS:
                if _is_true_sentence_end_on_word(s, stripped[-1]):
                    s = stripped[:-1].rstrip(".?!。？！") + quote_tail
        else:
            if stripped and stripped[-1] in CLAUSE_PUNCT_CHARS:
                stripped = stripped[:-1]
            if stripped and stripped[-1] not in SENTENCE_END_CHARS:
                s = stripped + end_punct + quote_tail
            elif not stripped:
                s = end_punct
        out[i] = s
    return out


def _render_sentences_from_spans(
    final_spans: list[tuple[int, int]],
    cleaned_words: list[str],
    global_punct_by_word: dict[int, str],
    joiner: str,
) -> tuple[list[str], list[tuple[int, int]]]:
    default_period = "。" if joiner == "" else "."
    for _, e_idx in final_spans:
        if (
            e_idx not in global_punct_by_word
            or global_punct_by_word[e_idx] in CLAUSE_PUNCT_CHARS
        ):
            w_end_str = cleaned_words[e_idx].rstrip(_TRAILING_QUOTE_CHARS)
            global_punct_by_word[e_idx] = (
                w_end_str[-1]
                if (w_end_str and w_end_str[-1] in SENTENCE_END_CHARS)
                else default_period
            )

    sentences: list[str] = []
    kept_spans: list[tuple[int, int]] = []
    for s_idx, e_idx in final_spans:
        slice_words = list(cleaned_words[s_idx : e_idx + 1])
        if not slice_words:
            continue
        for rel_i in range(len(slice_words) - 1):
            p_ch = global_punct_by_word.get(s_idx + rel_i, "")
            if p_ch and p_ch in CLAUSE_PUNCT_CHARS:
                w_cur = slice_words[rel_i]
                if w_cur and w_cur[-1] in SENTENCE_END_CHARS:
                    if _is_true_sentence_end_on_word(w_cur, w_cur[-1]):
                        w_cur = w_cur.rstrip(SENTENCE_END_CHARS)
                if (
                    w_cur
                    and w_cur[-1] not in CLAUSE_PUNCT_CHARS
                    and w_cur[-1] not in SENTENCE_END_CHARS
                ):
                    slice_words[rel_i] = w_cur + p_ch
        end_p = global_punct_by_word.get(e_idx, "")
        sent_end_punct = (
            end_p if (end_p and end_p in SENTENCE_END_CHARS) else default_period
        )
        sentence = (
            join_words(_normalize_slice_words(slice_words, sent_end_punct), joiner)
            .strip()
            .lstrip("，,。.:;；：?!？！… ")
        )
        if sentence:
            sentences.append(sentence)
            kept_spans.append((s_idx, e_idx + 1))
    return sentences, kept_spans

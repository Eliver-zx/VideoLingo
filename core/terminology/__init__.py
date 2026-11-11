from __future__ import annotations

from .candidates import _retrieve_rare_token_candidates
from .custom_terms import (
    _CUSTOM_TERMS_CACHE,
    TerminologyConfig,
    active_glossary_domain,
    load_custom_terms_from_config,
    should_load_domain_glossary,
)
from .indices import (
    _KB_INDEX_CACHE,
    _TERM_INDEX_CACHE,
    get_kb_entries,
    get_kb_index,
    get_term_index,
)
from .lexical import (
    _CJK_RE,
    _EXTRA_COMMON_WORDS,
    _KB_STOP_TOKENS,
    _SIGNIFICANT_SYMBOL_RE,
    _UNICODE_SEP_RE,
    _WORD_RE,
    _char_sim,
    _fold_ascii,
    _is_standard_english,
    _levenshtein_fast,
    _phonetic_skeleton,
    _span_sim,
    _stem_word,
    _tokenize_and_normalize,
    _tokenize_surface,
    _word_sim,
)
from .matcher import (
    SurfaceRule,
    TermHit,
    _build_term_matcher,
    _ensure_kb_matcher,
    _match_terms_in_text,
    _scan_text_for_kb_terms,
    _scan_text_with_covered_tokens,
    format_term_hits_for_prompt,
)
from .merge import merge_terminology_results
from .ngram_index import UnifiedNgramKBIndex, _char_phonetic_ngrams
from .normalize import (
    _coerce_bool,
    _extract_generic_kb_pairs,
    _normalize_aliases,
    _normalize_kb_entries,
)
from .shorthands import (
    _derive_in_video_shorthands,
    resolve_ambiguous_kb_candidates_with_llm,
)

__all__ = [
    "SurfaceRule",
    "TermHit",
    "TerminologyConfig",
    "UnifiedNgramKBIndex",
    "_CJK_RE",
    "_CUSTOM_TERMS_CACHE",
    "_EXTRA_COMMON_WORDS",
    "_KB_INDEX_CACHE",
    "_KB_STOP_TOKENS",
    "_SIGNIFICANT_SYMBOL_RE",
    "_TERM_INDEX_CACHE",
    "_UNICODE_SEP_RE",
    "_WORD_RE",
    "_build_term_matcher",
    "_char_phonetic_ngrams",
    "_char_sim",
    "_coerce_bool",
    "_derive_in_video_shorthands",
    "_ensure_kb_matcher",
    "_extract_generic_kb_pairs",
    "_fold_ascii",
    "_is_standard_english",
    "_levenshtein_fast",
    "_match_terms_in_text",
    "_normalize_aliases",
    "_normalize_kb_entries",
    "_phonetic_skeleton",
    "_retrieve_rare_token_candidates",
    "_scan_text_for_kb_terms",
    "_scan_text_with_covered_tokens",
    "_span_sim",
    "_stem_word",
    "_tokenize_and_normalize",
    "_tokenize_surface",
    "_word_sim",
    "active_glossary_domain",
    "format_term_hits_for_prompt",
    "get_kb_entries",
    "get_kb_index",
    "get_term_index",
    "load_custom_terms_from_config",
    "merge_terminology_results",
    "resolve_ambiguous_kb_candidates_with_llm",
    "should_load_domain_glossary",
]

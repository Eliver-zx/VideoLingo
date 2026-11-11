"""Cohesive subtitle splitting helpers for Stage 5 (`core._5_split_sub`)."""

from core.subtitle_split.clause_dp import (
    _merge_tiny_clauses,
    _partition_clauses_dp,
    _subsplit_overlong_cjk_clause,
    _try_local_clause_split,
)
from core.subtitle_split.llm_batch import (
    _apply_br_split_to_sentence,
    _batch_split_sentences_llm,
    _call_split_sentence,
)
from core.subtitle_split.metrics import (
    _BREAK_CONJUNCTIONS,
    _CLAUSE_PUNCT,
    _CLAUSE_SPLIT_RE,
    _LEADING_PUNCT_RE,
    _normalize_book_marks,
    calc_len,
    needs_subtitle_split,
)
from core.subtitle_split.ratio_align import (
    _find_spaced_cut,
    align_source_by_ratio,
    align_subs,
    get_target_parts,
)
from core.subtitle_split.word_span import (
    _align_part_end_words,
    _clean_alnum,
    _partition_word_span_for_cues,
    build_attempt_line_gaps,
    compute_utterance_word_gaps,
)

__all__ = [
    "_BREAK_CONJUNCTIONS",
    "_CLAUSE_PUNCT",
    "_CLAUSE_SPLIT_RE",
    "_LEADING_PUNCT_RE",
    "_align_part_end_words",
    "_apply_br_split_to_sentence",
    "_batch_split_sentences_llm",
    "_call_split_sentence",
    "_clean_alnum",
    "_find_spaced_cut",
    "_merge_tiny_clauses",
    "_normalize_book_marks",
    "_partition_clauses_dp",
    "_partition_word_span_for_cues",
    "_subsplit_overlong_cjk_clause",
    "_try_local_clause_split",
    "align_source_by_ratio",
    "align_subs",
    "build_attempt_line_gaps",
    "calc_len",
    "compute_utterance_word_gaps",
    "get_target_parts",
    "needs_subtitle_split",
]

import concurrent.futures
from typing import Any

from core import nlp_split as _nlp_split
from core.concurrency import ContextThreadPoolExecutor
from core.nlp_split import (
    CHUNK_WORDS,
    OVERLAP_WORDS,
    SENTENCE_END_CHARS,
    _clean_word,
    _merge_standalone_punct_tokens,
    _prepare_word_inputs,
    _protect_multiword_alias_spans,
    _render_sentences_from_spans,
    _resolve_utterance_spans,
    _restore_window_punctuation,
    _split_atomic_tokens,
    _split_long_span_at_clauses,
    _stitch_incomplete_clause_spans,
    _sync_word_dicts,
    _verify_word_alignment,
    build_overlapping_windows,
)
from core.utils import (
    ask_gpt,
    check_cancel,
    get_joiner,
    get_source_language,
    join_words,
    load_key_or,
    report_progress,
)
from core.workspace import get_active_workspace

__all__ = [
    "CHUNK_WORDS",
    "OVERLAP_WORDS",
    "SENTENCE_END_CHARS",
    "_build_utterances_from_sentences",
    "_clean_word",
    "_merge_standalone_punct_tokens",
    "_protect_multiword_alias_spans",
    "_restore_window_punctuation",
    "_split_atomic_tokens",
    "_split_long_span_at_clauses",
    "_stitch_incomplete_clause_spans",
    "_verify_word_alignment",
    "ask_gpt",
    "build_overlapping_windows",
    "get_active_workspace",
    "get_source_language",
    "join_words",
    "load_key_or",
    "restore_and_split_words",
    "split_by_spacy",
]


def __getattr__(name: str) -> Any:
    if hasattr(_nlp_split, name):
        return getattr(_nlp_split, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def restore_and_split_words(
    words: list[str] | list[dict[str, Any]],
    language: str | None = None,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    word_dicts: list[dict[str, Any]] | None = None,
    terms: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Restore punctuation across ``words`` in overlapping windows and split into sentences."""
    if not words:
        return []

    cleaned_words, active_dicts, caller_dicts, word_gaps = _prepare_word_inputs(
        words, word_dicts
    )
    if not cleaned_words:
        return []

    lang = language or get_source_language()
    joiner = get_joiner(lang)
    n_words = len(cleaned_words)
    windows = build_overlapping_windows(
        n_words, chunk_words=chunk_words, overlap_words=overlap_words
    )
    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))
    global_end_indices: set[int] = set()
    global_punct_by_word: dict[int, str] = {}
    total_win = max(1, len(windows))
    report_progress(
        0.05,
        f"标点恢复 0/{total_win}",
        f"正在使用 LLM 恢复标点与句子边界（共 {total_win} 个滑动窗口）",
        tag="nlp",
        label="NLP",
    )

    with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_win = {
            executor.submit(
                _restore_window_punctuation,
                idx,
                cleaned_words[w_start:w_end],
                joiner,
                lang,
                word_gaps[w_start:w_end] if word_gaps is not None else None,
            ): (idx, w_start, w_end, own_start, own_end)
            for idx, (w_start, w_end, own_start, own_end) in enumerate(windows)
        }
        try:
            for done_win, future in enumerate(
                concurrent.futures.as_completed(future_to_win), 1
            ):
                check_cancel()
                win_idx, w_start, _w_end, own_start, own_end = future_to_win[future]
                res = future.result()
                report_progress(
                    0.05 + 0.85 * (done_win / total_win),
                    f"标点恢复 {done_win}/{total_win}",
                    f"完成标点恢复窗口 #{win_idx + 1:02d} ({done_win}/{total_win})",
                    tag="nlp",
                    label="NLP",
                )
                local_ends, local_punct = res if isinstance(res, tuple) else (res, {})
                for local_idx, p_ch in local_punct.items():
                    if own_start <= (g_idx := w_start + local_idx) < own_end:
                        global_punct_by_word[g_idx] = p_ch
                for local_idx in local_ends:
                    if own_start <= (g_idx := w_start + local_idx) < own_end:
                        global_end_indices.add(g_idx)
        except BaseException:
            for f in future_to_win:
                f.cancel()
            raise

    _protect_multiword_alias_spans(
        cleaned_words, global_end_indices, global_punct_by_word, extra_terms=terms
    )
    global_end_indices.add(n_words - 1)

    raw_spans: list[tuple[int, int]] = []
    start_idx = 0
    for end_idx in sorted(idx for idx in global_end_indices if 0 <= idx < n_words):
        if end_idx >= start_idx:
            raw_spans.append((start_idx, end_idx))
            start_idx = end_idx + 1

    raw_spans = _stitch_incomplete_clause_spans(
        raw_spans, cleaned_words, global_punct_by_word, word_gaps
    )
    final_spans = [
        sub_span
        for s_idx, e_idx in raw_spans
        for sub_span in _split_long_span_at_clauses(
            s_idx, e_idx, cleaned_words, global_punct_by_word, active_dicts, word_gaps
        )
    ]
    sentences, kept_spans = _render_sentences_from_spans(
        final_spans, cleaned_words, global_punct_by_word, joiner
    )
    _sync_word_dicts(active_dicts, caller_dicts, global_punct_by_word, kept_spans)
    return sentences


def _build_utterances_from_sentences(
    sentences: list[str],
    word_dicts: list[dict[str, Any]],
    joiner: str,
    spans: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Build ``Utterance`` records directly from precomputed ``spans`` or token alignment."""
    if not sentences or not word_dicts:
        return []

    n_words = len(word_dicts)
    resolved_spans = _resolve_utterance_spans(sentences, word_dicts, spans=spans)
    utterances: list[dict[str, Any]] = []
    for u_idx, (sent, (w_start, w_end)) in enumerate(zip(sentences, resolved_spans)):
        w_start = max(0, min(int(w_start), n_words - 1))
        w_end = max(w_start + 1, min(int(w_end), n_words))
        start_sec = round(float(word_dicts[w_start].get("start", 0.0)), 3)
        end_sec = round(float(word_dicts[w_end - 1].get("end", start_sec)), 3)
        gap_after = (
            round(max(0.0, float(word_dicts[w_end].get("start", end_sec)) - end_sec), 3)
            if w_end < n_words
            else 0.0
        )
        raw_tokens = [
            str(w.get("text", w.get("word", ""))).strip().strip(',.;:?!，。；：？！"“”')
            for w in word_dicts[w_start:w_end]
        ]
        src_raw = join_words([t for t in raw_tokens if t], joiner).strip()
        utterances.append(
            {
                "id": u_idx,
                "word_span": [w_start, w_end],
                "start": start_sec,
                "end": end_sec,
                "gap_after": gap_after,
                "src_raw": src_raw,
                "src_punct": sent,
                "src": sent,
            }
        )
    return utterances


def split_by_spacy() -> None:
    ws = get_active_workspace()
    if ws.has_utterances():
        report_progress(
            1.0,
            "复用已断句结果",
            "工作区已存在断句记录，跳过标点与断句阶段",
            tag="nlp",
            label="NLP",
        )
        return

    language = get_source_language()
    joiner = get_joiner(language)
    report_progress(
        0.02,
        f"准备标点恢复 ({language})…",
        f"开始执行 LLM 标点恢复与语义断句 (语言: {language})",
        tag="nlp",
        label="NLP",
    )

    word_dicts: list[dict[str, Any]] = ws.load_words()
    words = [str(w.get("text", "")).strip('"').strip() for w in word_dicts]
    sentences = restore_and_split_words(words, language=language, word_dicts=word_dicts)
    spans = (
        word_dicts[0].pop("_sentence_spans", None)
        if word_dicts and isinstance(word_dicts[0].get("_sentence_spans"), list)
        else None
    )
    ws.save_words(word_dicts)
    ws.save_utterances(
        _build_utterances_from_sentences(sentences, word_dicts, joiner, spans=spans)
    )
    report_progress(
        1.0,
        f"共切分 {len(sentences)} 句",
        f"标点恢复与断句完成，共生成 {len(sentences)} 条语句",
        tag="nlp",
        label="NLP",
    )


if __name__ == "__main__":
    split_by_spacy()

from __future__ import annotations

from typing import Any

from core.concurrency import ContextThreadPoolExecutor
from core.prompts import get_summary_prompt
from core.terminology import (
    SurfaceRule as SurfaceRule,
    TermHit as TermHit,
    TerminologyConfig,
    UnifiedNgramKBIndex as UnifiedNgramKBIndex,
    _CJK_RE,
    _EXTRA_COMMON_WORDS as _EXTRA_COMMON_WORDS,
    _KB_INDEX_CACHE as _KB_INDEX_CACHE,
    _KB_STOP_TOKENS as _KB_STOP_TOKENS,
    _SIGNIFICANT_SYMBOL_RE as _SIGNIFICANT_SYMBOL_RE,
    _TERM_INDEX_CACHE as _TERM_INDEX_CACHE,
    _UNICODE_SEP_RE as _UNICODE_SEP_RE,
    _WORD_RE as _WORD_RE,
    _build_term_matcher as _build_term_matcher,
    _char_sim as _char_sim,
    _derive_in_video_shorthands as _derive_in_video_shorthands,
    _ensure_kb_matcher as _ensure_kb_matcher,
    _extract_generic_kb_pairs as _extract_generic_kb_pairs,
    _fold_ascii as _fold_ascii,
    _is_standard_english as _is_standard_english,
    _levenshtein_fast as _levenshtein_fast,
    _match_terms_in_text,
    _normalize_aliases,
    _normalize_kb_entries as _normalize_kb_entries,
    _phonetic_skeleton as _phonetic_skeleton,
    _retrieve_rare_token_candidates,
    _scan_text_for_kb_terms as _scan_text_for_kb_terms,
    _scan_text_with_covered_tokens,
    _span_sim as _span_sim,
    _stem_word as _stem_word,
    _tokenize_and_normalize as _tokenize_and_normalize,
    _tokenize_surface as _tokenize_surface,
    _word_sim as _word_sim,
    active_glossary_domain as active_glossary_domain,
    format_term_hits_for_prompt,
    get_kb_index,
    get_term_index,
    load_custom_terms_from_config,
    merge_terminology_results,
    resolve_ambiguous_kb_candidates_with_llm,
    should_load_domain_glossary,
)
from core.utils import (
    ask_gpt,
    check_cancel,
    check_file_exists,
    load_key_or,
    report_progress,
)
from core.workspace import GLOBAL_GLOSSARY_CSV, get_active_workspace

CUSTOM_TERMS_PATH = "custom_terms.xlsx"
OPTIONAL_CSV_TERMS_PATH = "custom_terms.csv"
OPTIONAL_JSON_TERMS_PATH = "term-map.json"
GLOBAL_GLOSSARY_PATH = str(GLOBAL_GLOSSARY_CSV)
SUMMARY_CHUNK_WEIGHTED_CHARS = 8000


def _runtime_config() -> TerminologyConfig:
    return TerminologyConfig(
        custom_terms_path=CUSTOM_TERMS_PATH,
        optional_csv_terms_path=OPTIONAL_CSV_TERMS_PATH,
        optional_json_terms_path=OPTIONAL_JSON_TERMS_PATH,
        global_glossary_path=GLOBAL_GLOSSARY_PATH,
        load_key_or_fn=load_key_or,
    )


def _should_load_domain_glossary() -> bool:
    return should_load_domain_glossary(_runtime_config())


def _active_glossary_domain() -> str:
    return active_glossary_domain(_runtime_config())


def load_custom_terms() -> list[dict[str, Any]]:
    return load_custom_terms_from_config(_runtime_config())


def _get_kb_index(custom_terms_list: list[dict[str, Any]]) -> dict[str, Any]:
    return get_kb_index(custom_terms_list, _runtime_config())


def _get_term_index() -> dict[str, Any]:
    return get_term_index(_runtime_config())


def _weighted_char_len(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    return cjk * 2 + (len(text) - cjk)


def split_summary_chunks(max_weighted_chars: int | None = None) -> list[str]:
    limit = max_weighted_chars or SUMMARY_CHUNK_WEIGHTED_CHARS
    utts = get_active_workspace().load_utterances()
    lines = [
        str(u.get("src_punct") or u.get("src") or "").strip()
        for u in utts
        if str(u.get("src_punct") or u.get("src") or "").strip()
    ]
    if not lines:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in lines:
        wlen = _weighted_char_len(line)
        if buf and buf_len + wlen > limit:
            chunks.append("\n".join(buf))
            buf, buf_len = [line], wlen
        else:
            buf.append(line)
            buf_len += wlen
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def combine_chunks() -> str:
    return "\n\n".join(split_summary_chunks())


def _valid_summary(response_data: Any) -> dict[str, str]:
    if not isinstance(response_data, dict):
        return {"status": "error", "message": "Response must be a JSON object"}
    terms = response_data.get("terms")
    if not isinstance(terms, list):
        return {"status": "error", "message": "Missing 'terms' list in JSON response"}
    for item in terms:
        if not isinstance(item, dict):
            return {"status": "error", "message": "Each term item must be an object"}
        if (
            not str(item.get("src") or "").strip()
            or not str(item.get("tgt") or "").strip()
        ):
            return {
                "status": "error",
                "message": "Term 'src' and 'tgt' must be non-empty",
            }
    return {"status": "success", "message": "Summary response validated"}


def _extract_chunk_terms(
    chunk_idx: int,
    chunk_text: str,
    chunk_kb_terms: list[dict[str, Any]],
    video_context_terms: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    check_cancel()
    prompt = get_summary_prompt(
        source_content=chunk_text,
        custom_terms_json={"terms": chunk_kb_terms},
        previously_extracted_terms=video_context_terms,
    )
    resp = ask_gpt(
        prompt, resp_type="json", valid_def=_valid_summary, log_title="summary"
    )
    return chunk_idx, (resp if isinstance(resp, dict) else {"theme": "", "terms": []})


def _resolve_ambiguous_kb_candidates_with_llm(
    unresolved_items: list[dict[str, Any]],
) -> dict[str, str]:
    return resolve_ambiguous_kb_candidates_with_llm(
        unresolved_items, ask_gpt_fn=ask_gpt, load_key_or_fn=load_key_or
    )


def _merge_terminology_results(
    chunk_results: list[tuple[int, dict[str, Any]]],
    custom_terms_list: list[dict[str, Any]],
    kb_data: dict[str, Any] | None = None,
    full_text: str = "",
    run_llm_disambiguation: bool = False,
    *,
    precomputed_scanned_terms: list[dict[str, Any]] | None = None,
    precomputed_covered_tokens: set[str] | None = None,
) -> dict[str, Any]:
    return merge_terminology_results(
        chunk_results,
        custom_terms_list,
        kb_data=kb_data,
        full_text=full_text,
        run_llm_disambiguation=run_llm_disambiguation,
        precomputed_scanned_terms=precomputed_scanned_terms,
        precomputed_covered_tokens=precomputed_covered_tokens,
        ask_gpt_fn=ask_gpt,
        load_key_or_fn=load_key_or,
    )


@check_file_exists("summarize")
def get_summary() -> None:
    ws = get_active_workspace()
    custom_terms_list = load_custom_terms()
    kb_data = _get_kb_index(custom_terms_list)
    text_chunks = split_summary_chunks()
    if not text_chunks:
        ws.save_terms("", [])
        return

    full_text = "\n".join(text_chunks)
    video_kb_terms, full_covered_tokens = _scan_text_with_covered_tokens(
        full_text, kb_data
    )
    total_chunks = len(text_chunks)
    max_workers = max(1, int(load_key_or("max_workers", 4) or 4))

    report_progress(
        0.1,
        f"术语提取 0/{total_chunks} 批",
        f"开始并行提取术语与摘要（共 {total_chunks} 批，全片预命中 {len(video_kb_terms)} 条权威术语）",
        tag="llm",
        label="TERMS",
    )

    chunk_results: list[tuple[int, dict[str, Any]]] = []
    with ContextThreadPoolExecutor(
        max_workers=min(max_workers, total_chunks)
    ) as executor:
        futures = []
        for idx, chunk in enumerate(text_chunks):
            chunk_kb, chunk_covered = _scan_text_with_covered_tokens(chunk, kb_data)
            chunk_srcs = {t["src"].lower() for t in chunk_kb}
            rare_cands = _retrieve_rare_token_candidates(
                chunk, kb_data, covered_tokens=chunk_covered
            )
            seen_cand_srcs = set(chunk_srcs)
            chunk_candidates = list(chunk_kb)
            for rc in rare_cands:
                rl = rc["src"].lower()
                if rl not in seen_cand_srcs:
                    seen_cand_srcs.add(rl)
                    chunk_candidates.append(rc)
                else:
                    for existing in chunk_candidates:
                        if existing["src"].lower() == rl:
                            existing["aliases"] = _normalize_aliases(
                                [
                                    *(existing.get("aliases") or []),
                                    *(rc.get("aliases") or []),
                                ],
                                existing["src"],
                            )
            context_kb = [
                t for t in video_kb_terms if t["src"].lower() not in chunk_srcs
            ]
            futures.append(
                executor.submit(
                    _extract_chunk_terms, idx, chunk, chunk_candidates, context_kb
                )
            )
        for done_count, fut in enumerate(futures, 1):
            check_cancel()
            chunk_results.append(fut.result())
            report_progress(
                0.1 + 0.8 * (done_count / total_chunks),
                f"术语提取 {done_count}/{total_chunks} 批",
                f"已完成第 {done_count}/{total_chunks} 批术语提取",
                tag="llm",
                label="TERMS",
            )

    merged = _merge_terminology_results(
        chunk_results,
        custom_terms_list,
        kb_data=kb_data,
        full_text=full_text,
        run_llm_disambiguation=True,
        precomputed_scanned_terms=video_kb_terms,
        precomputed_covered_tokens=full_covered_tokens,
    )
    ws.save_terms(merged.get("theme", ""), merged.get("terms", []))
    report_progress(
        1.0,
        f"术语提取完成（共 {len(merged.get('terms', []))} 条）",
        f"术语提取完成，共汇总 {len(merged.get('terms', []))} 条术语",
        tag="llm",
        label="TERMS",
    )


def search_things_to_note_in_prompt(sentence: str) -> str | None:
    if not sentence or not sentence.strip():
        return None
    idx_data = _get_term_index()
    hits = _match_terms_in_text(sentence, idx_data["matcher"])
    lines = format_term_hits_for_prompt(hits)
    return "\n".join(lines) if lines else None

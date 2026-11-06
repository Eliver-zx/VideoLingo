import concurrent.futures
import hashlib
import json
import re
import time
import pandas as pd
from core.concurrency import ContextThreadPoolExecutor
from core.translate_lines import translate_lines
from core._4_1_summarize import (
    _KB_CONVERSATIONAL_STOPWORDS,
    _KB_STOP_TOKENS,
    _get_term_index,
    _is_grammatical_inflection_of,
    search_things_to_note_in_prompt,
)
from core._6_gen_sub import align_timestamp
from core.utils import (
    check_cancel,
    check_file_exists,
    load_key,
    load_key_or,
    log_event,
    report_progress,
)
from core.workspace import get_active_workspace


def _cand_regex_str(cand: str) -> str:
    if re.match(r"^[\w\s\.\-]+$", cand, flags=re.ASCII):
        toks = re.findall(r"\w+", cand, flags=re.ASCII)
        if len(toks) >= 2:
            sep = r"(?:[\s,\-–—]+|\s*\.\s*)+"
            trail_dot = r"\.?" if cand.endswith(".") else ""
            return rf"(?<!\w){sep.join(re.escape(t) for t in toks)}{trail_dot}(?!\w)"
        return rf"(?<!\w){re.escape(cand)}(?!\w)"
    return re.escape(cand)


def apply_fix_source_terms(text: str, terms: list[dict]) -> str:
    """Rewrite ASR alias variants into canonical ``src`` when ``fix_source`` is enabled on a term."""
    if not text or not terms:
        return text
    canonical_lows = {
        str(e.get("src") or "").strip().lower()
        for e in terms
        if isinstance(e, dict) and str(e.get("src") or "").strip()
    }
    out = text
    for entry in terms:
        if not isinstance(entry, dict):
            continue
        is_task_alias = (
            bool(entry.get("aliases"))
            and str(entry.get("source") or "task") == "task"
            and entry.get("fix_source") is not False
        )
        if not (entry.get("fix_source") or is_task_alias):
            continue
        canonical = str(entry.get("src") or "").strip()
        if not canonical:
            continue
        raw_aliases = entry.get("aliases") or []
        candidates: list[str] = []
        for a in raw_aliases:
            a_str = str(a).strip()
            if not a_str:
                continue
            a_low = a_str.lower()
            if (
                a_low != canonical.lower()
                and a_low not in canonical_lows
                and a_low not in _KB_STOP_TOKENS
                and a_low not in _KB_CONVERSATIONAL_STOPWORDS
                and not _is_grammatical_inflection_of(a_str, canonical)
            ):
                candidates.append(a_str)
        if not candidates and (raw_aliases or not entry.get("fix_source")):
            continue
        all_targets = sorted({*candidates, canonical}, key=len, reverse=True)
        pattern = re.compile(
            "|".join(f"(?:{_cand_regex_str(c)})" for c in all_targets if c),
            flags=re.IGNORECASE,
        )
        out = pattern.sub(canonical, out)
    return out


# Function to split text into chunks
def split_chunks_by_chars(chunk_size: int | None = None, max_i: int = 20):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    ws = get_active_workspace()
    utts = ws.load_utterances()
    sentences = [
        str(u.get("src") or "").strip() for u in utts if str(u.get("src") or "").strip()
    ]

    if not sentences:
        return []

    terms = _get_term_index().get("terms", [])
    has_fix = any(
        isinstance(t, dict)
        and (
            t.get("fix_source")
            or (t.get("aliases") and str(t.get("source") or "task") == "task")
        )
        for t in terms
    )
    has_prev_punct = any(isinstance(u, dict) and u.get("src_punct") for u in utts)
    if has_fix or has_prev_punct:
        base_sentences = [
            str(u.get("src_punct") or u.get("src") or "").strip()
            for u in utts
            if str(u.get("src") or "").strip()
        ]
        new_sentences = [apply_fix_source_terms(s, terms) for s in base_sentences]
        if new_sentences != sentences or (has_fix and not has_prev_punct):
            orig_sentences = list(sentences)
            sentences = new_sentences
            if utts and len(utts) == len(sentences):
                for idx, u in enumerate(utts):
                    u.setdefault("src_raw", orig_sentences[idx])
                    u.setdefault("src_punct", base_sentences[idx])
                    u["src"] = sentences[idx]
                ws.save_utterances(utts, mark_translated=False)

    chunks = []
    chunk = ""
    sentence_count = 0
    for sentence in sentences:
        exceeds_chars = chunk_size is not None and (
            len(chunk) + len(sentence + "\n") > chunk_size
        )
        if exceeds_chars or sentence_count == max_i:
            if (
                chunk.strip()
            ):  # a first sentence longer than chunk_size leaves nothing to flush
                chunks.append(chunk.strip())
            chunk = sentence + "\n"
            sentence_count = 1
        else:
            chunk += sentence + "\n"
            sentence_count += 1
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks


# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return (
        None if chunk_index == 0 else chunks[chunk_index - 1].split("\n")[-4:]
    )  # Get last 4 lines


def get_after_content(chunks, chunk_index):
    return (
        None
        if chunk_index == len(chunks) - 1
        else chunks[chunk_index + 1].split("\n")[:3]
    )  # Get first 3 lines


class ChunkTranslateResult(tuple):
    """3-tuple (index, english_result, translation) carrying `.direct_lines`."""

    direct_lines: list[str] | None

    def __new__(
        cls,
        index: int,
        english_result: str,
        translation: str,
        direct_lines: list[str] | None = None,
    ):
        obj = super().__new__(cls, (index, english_result, translation))
        obj.direct_lines = direct_lines
        return obj


# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i):
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    res = translate_lines(
        chunk,
        previous_content_prompt,
        after_content_prompt,
        things_to_note_prompt,
        theme_prompt,
        i,
    )
    translation, english_result = res
    direct_lines = getattr(res, "direct_lines", None)
    return ChunkTranslateResult(i, english_result, translation, direct_lines)


def _translation_cache_fingerprint() -> str:
    """Compute a SHA-256 fingerprint of translation settings and merged terminology content."""
    h = hashlib.sha256()
    for key in (
        "api.model",
        "api.reasoning_effort",
        "target_language",
        "translation_style",
        "reflect_translate",
        "glossary_domain",
    ):
        val = str(load_key_or(key, "") or "")
        h.update(f"{key}={val}\n".encode("utf-8"))
    ws = get_active_workspace()
    if ws.has_terms():
        h.update(
            json.dumps(ws.load_terms(), sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )
    try:
        merged_terms = _get_term_index().get("terms", [])
        if merged_terms:
            h.update(
                json.dumps(merged_terms, sort_keys=True, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
    except Exception:
        pass
    return h.hexdigest()[:16]


def _load_cached_raw_translation(
    expected_sources: list[str], fingerprint: str = ""
) -> pd.DataFrame | None:
    """Return cached raw translation DataFrame if it exists and matches ``expected_sources`` and ``fingerprint``."""
    if not fingerprint:
        return None
    ws = get_active_workspace()
    db_cached = ws.load_translation_cache(fingerprint, expected_sources)
    if db_cached is not None:
        return pd.DataFrame({"Source": expected_sources, "Translation": db_cached})
    return None


def _flush_chunk_translation_live(
    ws,
    utterances: list[dict],
    non_empty_indices: list[int],
    chunk_offsets: list[tuple[int, list[str]]],
    res_item: ChunkTranslateResult,
    last_flush_ts: float,
) -> float:
    chunk_idx, _, chunk_trans = res_item
    if chunk_idx < 0 or chunk_idx >= len(chunk_offsets):
        return last_flush_ts
    offset, chunk_lines = chunk_offsets[chunk_idx]
    trans_lines = chunk_trans.split("\n")
    if len(trans_lines) != len(chunk_lines):
        return last_flush_ts
    direct_chunk = getattr(res_item, "direct_lines", None)
    for j, tgt_raw in enumerate(trans_lines):
        utt_pos = offset + j
        if utt_pos >= len(non_empty_indices):
            break
        utt = utterances[non_empty_indices[utt_pos]]
        tgt_str = str(tgt_raw).strip()
        dir_str = (
            str(direct_chunk[j]).strip()
            if direct_chunk and j < len(direct_chunk)
            else tgt_str
        )
        utt["direct"] = dir_str
        utt["tgt"] = tgt_str
        utt["tts_text"] = tgt_str
    now = time.monotonic()
    if now - last_flush_ts >= 1.5:
        ws.save_utterances(utterances, mark_translated=False)
        return now
    return last_flush_ts


# 🚀 Main function to translate all chunks
@check_file_exists("translate")
def translate_all():
    chunks = split_chunks_by_chars(max_i=20)
    expected_sources = [
        line for chunk in chunks for line in chunk.split("\n") if line.strip()
    ]
    fingerprint = _translation_cache_fingerprint()
    ws = get_active_workspace()

    df_translate = _load_cached_raw_translation(
        expected_sources, fingerprint=fingerprint
    )
    direct_lines_all: list[str] = []
    if df_translate is not None:
        log_event(
            f"命中字幕翻译缓存 ({len(expected_sources)} 句)",
            tag="llm",
            label="CACHE",
        )
    else:
        term_data = ws.load_terms() if ws.has_terms() else {}
        theme_prompt = (
            term_data.get("theme") or term_data.get("topic") or term_data.get("summary")
        )

        live_utts = ws.load_utterances()
        non_empty_indices = [
            idx for idx, u in enumerate(live_utts) if str(u.get("src") or "").strip()
        ]
        chunk_offsets: list[tuple[int, list[str]]] = []
        running_offset = 0
        for ch in chunks:
            ch_lines = ch.split("\n")
            chunk_offsets.append((running_offset, ch_lines))
            running_offset += len(ch_lines)
        can_live_flush = bool(live_utts) and len(non_empty_indices) == running_offset
        last_flush_ts = 0.0

        total_chunks = max(1, len(chunks))
        report_progress(
            0.05,
            f"字幕翻译 0/{len(chunks)} 批",
            f"开始并发翻译字幕（共 {len(chunks)} 批）",
            tag="llm",
            label="TRANS",
        )
        with ContextThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    translate_chunk, chunk, chunks, theme_prompt, i
                )
                futures.append(future)
            results = []
            try:
                for future in concurrent.futures.as_completed(futures):
                    check_cancel()
                    res_item = future.result()
                    results.append(res_item)
                    if can_live_flush:
                        last_flush_ts = _flush_chunk_translation_live(
                            ws,
                            live_utts,
                            non_empty_indices,
                            chunk_offsets,
                            res_item,
                            last_flush_ts,
                        )
                    done_cnt = len(results)
                    report_progress(
                        0.05 + 0.90 * (done_cnt / total_chunks),
                        f"字幕翻译 {done_cnt}/{len(chunks)} 批",
                        f"完成翻译批次 #{res_item[0] + 1:02d} ({done_cnt}/{len(chunks)})",
                        tag="llm",
                        label="TRANS",
                    )
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

        results.sort(key=lambda x: x[0])  # Sort results based on original chunk index

        src_text, trans_text = [], []
        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.split("\n")
            src_text.extend(chunk_lines)
            _, _, chunk_trans = results[i]
            trans_lines = chunk_trans.split("\n")
            if len(trans_lines) != len(chunk_lines):
                raise ValueError(
                    f"Translation line count mismatch in chunk {i}: expected {len(chunk_lines)}, got {len(trans_lines)}"
                )
            trans_text.extend(trans_lines)
            direct_chunk = getattr(results[i], "direct_lines", None)
            if direct_chunk and len(direct_chunk) == len(chunk_lines):
                direct_lines_all.extend(direct_chunk)
            else:
                direct_lines_all.extend(trans_lines)

        df_translate = pd.DataFrame({"Source": src_text, "Translation": trans_text})
        ws.save_translation_cache(fingerprint, list(zip(src_text, trans_text)))

    utterances = ws.load_utterances()
    tr_list = df_translate["Translation"].tolist()
    if utterances and len(utterances) == len(tr_list):
        for idx, u in enumerate(utterances):
            tgt_str = str(tr_list[idx]).strip()
            dir_str = (
                direct_lines_all[idx]
                if idx < len(direct_lines_all)
                else str(u.get("direct") or tgt_str)
            )
            u["direct"] = dir_str
            u["tgt"] = tgt_str
            u["tts_text"] = tgt_str
        ws.save_utterances(utterances)
    else:
        word_records = ws.load_words()
        df_text = (
            pd.DataFrame(word_records)
            if word_records
            else pd.DataFrame(columns=["text", "start", "end"])
        )
        subtitle_output_configs = [("trans_subs_for_audio.srt", ["Translation"])]
        df_time = align_timestamp(
            df_text,
            df_translate,
            subtitle_output_configs,
            output_dir=None,
            for_display=False,
        )
        built_utts = []
        for idx, row in df_time.iterrows():
            ts = row.get("timestamp") or (0.0, 0.0)
            s_sec = (
                float(ts[0]) if isinstance(ts, (list, tuple)) and len(ts) >= 2 else 0.0
            )
            e_sec = (
                float(ts[1]) if isinstance(ts, (list, tuple)) and len(ts) >= 2 else 0.0
            )
            tgt_str = str(row.get("Translation") or "").strip()
            dir_str = (
                direct_lines_all[len(built_utts)]
                if len(built_utts) < len(direct_lines_all)
                else tgt_str
            )
            built_utts.append(
                {
                    "id": f"u_{len(built_utts):04d}",
                    "index": len(built_utts),
                    "word_span": [0, 0],
                    "start": round(s_sec, 3),
                    "end": round(e_sec, 3),
                    "gap_after": 0.0,
                    "src": str(row.get("Source") or "").strip(),
                    "direct": dir_str,
                    "tgt": tgt_str,
                    "tts_text": tgt_str,
                }
            )
        ws.save_utterances(built_utts)

    report_progress(
        1.0,
        f"已翻译 {len(tr_list)} 句字幕",
        f"字幕翻译完成并保存（共 {len(tr_list)} 句）",
        tag="llm",
        label="TRANS",
    )


if __name__ == "__main__":
    translate_all()

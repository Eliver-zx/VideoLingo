import concurrent.futures
import hashlib
import json
import os
import re
import pandas as pd
from core.concurrency import ContextThreadPoolExecutor
from core.translate_lines import translate_lines
from core._4_1_summarize import _get_term_index, search_things_to_note_in_prompt
from core._6_gen_sub import align_timestamp
from core.utils import check_cancel, check_file_exists, load_key, load_key_or
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from core.utils.models import (
    _2_CLEANED_CHUNKS,
    _3_2_SPLIT_BY_MEANING,
    _4_1_TERMINOLOGY,
    _4_2_TRANSLATION,
    resolve_task_path,
)
from core.workspace import workspace_for_path

console = Console()

_4_2_TRANSLATION_RAW = "output/log/translation_raw.csv"


def apply_fix_source_terms(text: str, terms: list[dict]) -> str:
    """Rewrite ASR alias variants into canonical ``src`` when ``fix_source`` is enabled on a term."""
    if not text or not terms:
        return text
    out = text
    for entry in terms:
        if not isinstance(entry, dict) or not entry.get("fix_source"):
            continue
        canonical = str(entry.get("src") or "").strip()
        if not canonical:
            continue
        raw_aliases = entry.get("aliases") or []
        candidates = [str(a).strip() for a in raw_aliases if str(a).strip()] + [
            canonical
        ]
        for cand in sorted(set(candidates), key=len, reverse=True):
            if re.match(r"^[\w\s\-]+$", cand, flags=re.ASCII):
                pattern = re.compile(
                    rf"(?<!\w){re.escape(cand)}(?!\w)", flags=re.IGNORECASE
                )
            else:
                pattern = re.compile(re.escape(cand), flags=re.IGNORECASE)
            out = pattern.sub(canonical, out)
    return out


# Function to split text into chunks
def split_chunks_by_chars(chunk_size=2400, max_i=15):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    meaning_path = resolve_task_path(_3_2_SPLIT_BY_MEANING)
    ws = workspace_for_path(meaning_path)
    utts = ws.load_utterances()
    if utts:
        sentences = [
            str(u.get("src") or "").strip()
            for u in utts
            if str(u.get("src") or "").strip()
        ]
    elif os.path.exists(meaning_path):
        with open(meaning_path, "r", encoding="utf-8") as file:
            sentences = [
                line.strip() for line in file.read().split("\n") if line.strip()
            ]
    else:
        sentences = []

    if not sentences:
        return []

    terms = _get_term_index().get("terms", [])
    if any(isinstance(t, dict) and t.get("fix_source") for t in terms):
        new_sentences = [apply_fix_source_terms(s, terms) for s in sentences]
        if new_sentences != sentences:
            orig_sentences = list(sentences)
            sentences = new_sentences
            if utts and len(utts) == len(sentences):
                for idx, u in enumerate(utts):
                    u.setdefault("src_raw", orig_sentences[idx])
                    u["src"] = sentences[idx]
                ws.save_utterances(utts)
            elif os.path.exists(meaning_path):
                with open(meaning_path, "w", encoding="utf-8") as wf:
                    wf.write("\n".join(sentences) + "\n")

    chunks = []
    chunk = ""
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + "\n") > chunk_size or sentence_count == max_i:
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
    prev_str = (
        "\n".join(previous_content_prompt)
        if isinstance(previous_content_prompt, list)
        else str(previous_content_prompt or "")
    )
    after_str = (
        "\n".join(after_content_prompt)
        if isinstance(after_content_prompt, list)
        else str(after_content_prompt or "")
    )
    search_scope = "\n".join(part for part in (chunk, prev_str, after_str) if part)
    things_to_note_prompt = search_things_to_note_in_prompt(search_scope)
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
    term_path = resolve_task_path(_4_1_TERMINOLOGY)
    ws = workspace_for_path(term_path)
    if ws.has_terms():
        h.update(
            json.dumps(ws.load_terms(), sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        )
    elif os.path.exists(term_path):
        try:
            with open(term_path, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
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
    raw_path = resolve_task_path(_4_2_TRANSLATION_RAW)
    ws = workspace_for_path(raw_path)
    if fingerprint:
        db_cached = ws.load_translation_cache(fingerprint, expected_sources)
        if db_cached is not None:
            return pd.DataFrame({"Source": expected_sources, "Translation": db_cached})
    if not os.path.exists(raw_path):
        return None
    try:
        df_cached = pd.read_csv(raw_path, encoding="utf-8-sig")
        if "Source" not in df_cached.columns or "Translation" not in df_cached.columns:
            return None
        if fingerprint:
            if "Fingerprint" not in df_cached.columns or len(df_cached) == 0:
                return None
            cached_fp = str(df_cached["Fingerprint"].iloc[0]).strip()
            if cached_fp != fingerprint:
                return None
        cached_src = [str(s).strip() for s in df_cached["Source"].fillna("")]
        target_src = [str(s).strip() for s in expected_sources]
        if cached_src == target_src and len(df_cached) == len(expected_sources):
            return pd.DataFrame(
                {
                    "Source": expected_sources,
                    "Translation": [
                        str(t) for t in df_cached["Translation"].fillna("")
                    ],
                }
            )
    except Exception:
        return None
    return None


# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    chunks = split_chunks_by_chars(chunk_size=2400, max_i=15)
    expected_sources = [
        line for chunk in chunks for line in chunk.split("\n") if line.strip()
    ]
    fingerprint = _translation_cache_fingerprint()
    term_path = resolve_task_path(_4_1_TERMINOLOGY)
    cleaned_path = resolve_task_path(_2_CLEANED_CHUNKS)
    out_path = resolve_task_path(_4_2_TRANSLATION)
    ws = workspace_for_path(out_path)

    df_translate = _load_cached_raw_translation(
        expected_sources, fingerprint=fingerprint
    )
    direct_lines_all: list[str] = []
    if df_translate is not None:
        console.print(
            f"[cyan]♻️ Reusing cached raw translations from `{ws.db_path}`[/cyan]"
        )
    else:
        if ws.has_terms():
            term_data = ws.load_terms()
        elif os.path.exists(term_path):
            with open(term_path, "r", encoding="utf-8") as file:
                term_data = json.load(file)
        else:
            term_data = {}
        theme_prompt = (
            term_data.get("theme") or term_data.get("topic") or term_data.get("summary")
        )

        # 🔄 Use concurrent execution for translation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
            with ContextThreadPoolExecutor(
                max_workers=load_key("max_workers")
            ) as executor:
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
                        results.append(future.result())
                        progress.update(task, advance=1)
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
        if word_records:
            df_text = pd.DataFrame(word_records)
        elif os.path.exists(cleaned_path):
            df_text = pd.read_excel(cleaned_path)
            df_text["text"] = df_text["text"].str.strip('"').str.strip()
        else:
            df_text = pd.DataFrame(columns=["text", "start", "end"])
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

    console.print(
        f"[bold green]✅ Translation completed and saved to `{ws.db_path}`.[/bold green]"
    )


if __name__ == "__main__":
    translate_all()

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import pandas as pd
from core.translate_lines import LAST_DIRECT_BY_INDEX, translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
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
from core.workspace import TaskWorkspace, get_active_workspace
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
        candidates = [str(a).strip() for a in raw_aliases if str(a).strip()] + [canonical]
        for cand in sorted(set(candidates), key=len, reverse=True):
            if re.match(r"^[\w\s\-]+$", cand, flags=re.ASCII):
                pattern = re.compile(rf"(?<!\w){re.escape(cand)}(?!\w)", flags=re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(cand), flags=re.IGNORECASE)
            out = pattern.sub(canonical, out)
    return out


# Function to split text into chunks
def split_chunks_by_chars(chunk_size=2400, max_i=15): 
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    meaning_path = resolve_task_path(_3_2_SPLIT_BY_MEANING)
    with open(meaning_path, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.read().split('\n') if line.strip()]

    if not sentences:
        return []

    term_path = resolve_task_path(_4_1_TERMINOLOGY)
    if os.path.isfile(term_path):
        try:
            with open(term_path, "r", encoding="utf-8") as tf:
                term_data = json.load(tf)
            terms = term_data.get("terms", []) if isinstance(term_data, dict) else []
            if any(isinstance(t, dict) and t.get("fix_source") for t in terms):
                new_sentences = [apply_fix_source_terms(s, terms) for s in sentences]
                if new_sentences != sentences:
                    sentences = new_sentences
                    with open(meaning_path, "w", encoding="utf-8") as wf:
                        wf.write("\n".join(sentences) + "\n")
                    ws = get_active_workspace()
                    if ws.utterances_path.is_file():
                        utts = ws.load_utterances()
                        if len(utts) == len(sentences):
                            for idx, u in enumerate(utts):
                                u["src"] = sentences[idx]
                            ws.save_utterances(utts)
        except Exception:
            pass

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            if chunk.strip():  # a first sentence longer than chunk_size leaves nothing to flush
                chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-4:] # Get last 4 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:3] # Get first 3 lines

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
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation


def _translation_cache_fingerprint() -> str:
    """Compute a SHA-256 fingerprint of translation settings and terminology content.

    Ensures cached raw translations are invalidated whenever ``api.model``,
    ``target_language``, ``translation_style``, ``reflect_translate``, or
    ``terminology.json`` changes.
    """
    h = hashlib.sha256()
    for key in ("api.model", "target_language", "translation_style", "reflect_translate"):
        val = str(load_key_or(key, "") or "")
        h.update(f"{key}={val}\n".encode("utf-8"))
    term_path = resolve_task_path(_4_1_TERMINOLOGY)
    if os.path.exists(term_path):
        try:
            with open(term_path, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _load_cached_raw_translation(expected_sources: list[str], fingerprint: str = "") -> pd.DataFrame | None:
    """Return cached raw translation DataFrame if it exists and matches ``expected_sources`` and ``fingerprint``."""
    raw_path = resolve_task_path(_4_2_TRANSLATION_RAW)
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
            return pd.DataFrame({
                "Source": expected_sources,
                "Translation": [str(t) for t in df_cached["Translation"].fillna("")],
            })
    except Exception:
        return None
    return None


# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    chunks = split_chunks_by_chars(chunk_size=2400, max_i=15)
    expected_sources = [line for chunk in chunks for line in chunk.split('\n') if line.strip()]
    fingerprint = _translation_cache_fingerprint()
    raw_path = resolve_task_path(_4_2_TRANSLATION_RAW)
    term_path = resolve_task_path(_4_1_TERMINOLOGY)
    cleaned_path = resolve_task_path(_2_CLEANED_CHUNKS)
    out_path = resolve_task_path(_4_2_TRANSLATION)

    df_translate = _load_cached_raw_translation(expected_sources, fingerprint=fingerprint)
    direct_lines_all: list[str] = []
    if df_translate is not None:
        console.print(f"[cyan]♻️ Reusing cached raw translations from `{raw_path}`[/cyan]")
    else:
        LAST_DIRECT_BY_INDEX.clear()
        with open(term_path, 'r', encoding='utf-8') as file:
            term_data = json.load(file)
            theme_prompt = term_data.get('theme') or term_data.get('topic') or term_data.get('summary')

        # 🔄 Use concurrent execution for translation
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
            with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
                futures = []
                for i, chunk in enumerate(chunks):
                    future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
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
        
        # 💾 Save results to lists and Excel file
        src_text, trans_text = [], []
        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.split('\n')
            src_text.extend(chunk_lines)
            _, _, chunk_trans = results[i]
            trans_lines = chunk_trans.split('\n')
            if len(trans_lines) != len(chunk_lines):
                raise ValueError(f"Translation line count mismatch in chunk {i}: expected {len(chunk_lines)}, got {len(trans_lines)}")
            trans_text.extend(trans_lines)
            direct_chunk = LAST_DIRECT_BY_INDEX.get(i)
            if direct_chunk and len(direct_chunk) == len(chunk_lines):
                direct_lines_all.extend(direct_chunk)
            else:
                direct_lines_all.extend(trans_lines)
        
        df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
        os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
        df_to_cache = df_translate.copy()
        df_to_cache['Fingerprint'] = fingerprint
        df_to_cache.to_csv(raw_path, index=False, encoding='utf-8-sig')

    # Update utterances.json with direct, tgt, and tts_text (preserving full punctuation)
    ws = (
        get_active_workspace()
        if str(_4_2_TRANSLATION).startswith("output/")
        else TaskWorkspace(
            Path(out_path).parent.parent
            if Path(out_path).parent.name == "log"
            else Path(out_path).parent
        )
    )
    if ws.utterances_path.is_file():
        utterances = ws.load_utterances()
        tr_list = df_translate["Translation"].tolist()
        for idx, u in enumerate(utterances):
            if idx < len(tr_list):
                tgt_str = str(tr_list[idx]).strip()
                dir_str = direct_lines_all[idx] if idx < len(direct_lines_all) else str(u.get("direct") or tgt_str)
                u["direct"] = dir_str
                u["tgt"] = tgt_str
                u["tts_text"] = tgt_str
        ws.save_utterances(utterances)

    df_text = pd.read_excel(cleaned_path)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_time.to_excel(out_path, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
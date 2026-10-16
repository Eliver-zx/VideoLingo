import pandas as pd
import json
import concurrent.futures
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from core.utils.models import *
console = Console()

_4_2_TRANSLATION_RAW = "output/log/translation_raw.csv"

# Function to split text into chunks
def split_chunks_by_chars(chunk_size=2400, max_i=15): 
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file.read().split('\n') if line.strip()]

    if not sentences:
        return []

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
    search_scope = "\n".join(previous_content_prompt) + "\n" + chunk if previous_content_prompt else chunk
    things_to_note_prompt = search_things_to_note_in_prompt(search_scope)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation


def _translation_cache_fingerprint() -> str:
    """Compute a SHA-256 fingerprint of translation settings and terminology content.

    Ensures cached raw translations are invalidated whenever ``api.model``,
    ``target_language``, ``translation_style``, ``reflect_translate``, or
    ``terminology.json`` changes.
    """
    import hashlib
    import os

    h = hashlib.sha256()
    for key in ("api.model", "target_language", "translation_style", "reflect_translate"):
        val = str(load_key_or(key, "") or "")
        h.update(f"{key}={val}\n".encode("utf-8"))
    if os.path.exists(_4_1_TERMINOLOGY):
        try:
            with open(_4_1_TERMINOLOGY, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:16]


def _load_cached_raw_translation(expected_sources: list[str], fingerprint: str = "") -> pd.DataFrame | None:
    """Return cached raw translation DataFrame if it exists and matches ``expected_sources`` and ``fingerprint``."""
    import os
    if not os.path.exists(_4_2_TRANSLATION_RAW):
        return None
    try:
        df_cached = pd.read_csv(_4_2_TRANSLATION_RAW, encoding="utf-8-sig")
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

    df_translate = _load_cached_raw_translation(expected_sources, fingerprint=fingerprint)
    if df_translate is not None:
        console.print(f"[cyan]♻️ Reusing cached raw translations from `{_4_2_TRANSLATION_RAW}`[/cyan]")
    else:
        with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
            term_data = json.load(file)
            theme_prompt = term_data.get('theme') or term_data.get('topic')

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
        
        df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
        import os
        os.makedirs(os.path.dirname(_4_2_TRANSLATION_RAW), exist_ok=True)
        df_to_cache = df_translate.copy()
        df_to_cache['Fingerprint'] = fingerprint
        df_to_cache.to_csv(_4_2_TRANSLATION_RAW, index=False, encoding='utf-8-sig')

    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
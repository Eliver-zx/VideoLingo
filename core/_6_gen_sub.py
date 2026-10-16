import pandas as pd
import os
import re
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def _levenshtein_fast(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein_fast(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]

def format_subtitle_text(x):
    """Format subtitle text: remove spaces between CJK and Latin/digits, replace CJK clause punctuation with spaces."""
    if pd.isna(x):
        return ''
    text = str(x)
    digit_pattern = r'[\d\uff10-\uff19]'
    text = re.sub(r'([\u4e00-\u9fff])\s+(' + digit_pattern + r')', r'\1\2', text)
    text = re.sub(r'(' + digit_pattern + r')\s+([\u4e00-\u9fff])', r'\1\2', text)
    text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s+([\u4e00-\u9fff])', r'\1\2', text)
    text = re.sub(r'[，。、？！：；]', ' ', text)
    return text.strip()

def get_sentence_timestamps(df_words, df_sentences):
    time_stamp_list = []
    
    # Build complete string and position mapping
    full_words_str = ''
    position_to_word_idx = {}
    word_starts = []
    word_ends = []
    
    for idx, word in enumerate(df_words['text']):
        word_str = '' if pd.isna(word) else str(word)
        clean_word = remove_punctuation(word_str.lower()).replace(" ", "")
        start_pos = len(full_words_str)
        full_words_str += clean_word
        end_pos = len(full_words_str)
        if end_pos > start_pos:
            word_starts.append(start_pos)
            word_ends.append(end_pos)
        for pos in range(start_pos, end_pos):
            position_to_word_idx[pos] = idx
    
    current_pos = 0
    for idx, sentence in df_sentences['Source'].items():
        sentence_str = '' if pd.isna(sentence) else str(sentence)
        clean_sentence = remove_punctuation(sentence_str.lower()).replace(" ", "")
        sentence_len = len(clean_sentence)

        # A line with nothing but punctuation has no words to match; place it after the loop
        if sentence_len == 0:
            time_stamp_list.append(None)
            continue

        match_found = False
        pos = full_words_str.find(clean_sentence, current_pos)
        if pos != -1:
            start_word_idx = position_to_word_idx[pos]
            end_word_idx = position_to_word_idx[pos + sentence_len - 1]
            time_stamp_list.append((
                float(df_words['start'][start_word_idx]),
                float(df_words['end'][end_word_idx])
            ))
            current_pos = pos + sentence_len
            match_found = True
        else:
            # Word-boundary fuzzy match fallback (tolerating up to 2 character edits)
            max_dist = 2
            win_low = max(0, current_pos - 8)
            win_high = min(len(full_words_str), current_pos + sentence_len + 16)
            cand_starts = [s for s in word_starts if win_low <= s <= min(win_high, current_pos + 12)]
            cand_ends = [e for e in word_ends if win_low < e <= win_high]
            best_span = None
            best_dist = max_dist + 1
            for s in cand_starts:
                for e in cand_ends:
                    if e <= s or abs((e - s) - sentence_len) > max_dist:
                        continue
                    d = _levenshtein_fast(full_words_str[s:e], clean_sentence)
                    if d < best_dist:
                        best_dist = d
                        best_span = (s, e)
            if best_span is not None and best_dist <= max_dist:
                s, e = best_span
                start_word_idx = position_to_word_idx[s]
                end_word_idx = position_to_word_idx[e - 1]
                time_stamp_list.append((
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                ))
                current_pos = e
                match_found = True
            
        if not match_found:
            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(clean_sentence, 
                          full_words_str[current_pos:current_pos+len(clean_sentence)])
            print("\nOriginal sentence:", df_sentences['Source'][idx])
            raise ValueError("❎ No match found for sentence.")

    # Punctuation-only lines fill the gap between their neighbours
    for i, stamp in enumerate(time_stamp_list):
        if stamp is None:
            start = time_stamp_list[i-1][1] if i > 0 else 0.0
            following = next((s for s in time_stamp_list[i+1:] if s is not None), None)
            time_stamp_list[i] = (start, max(start, following[0]) if following else start)

    return time_stamp_list

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # Remove gaps and fix overlaps 🕳️
    for i in range(len(df_trans_time)-1):
        curr_start, curr_end = df_trans_time.loc[i, 'timestamp']
        next_start, next_end = df_trans_time.loc[i+1, 'timestamp']
        delta_time = next_start - curr_end
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (curr_start, next_start)
        elif delta_time < 0:
            midpoint = max(curr_start, min(next_end, curr_end + delta_time / 2))
            df_trans_time.at[i, 'timestamp'] = (curr_start, midpoint)
            df_trans_time.at[i+1, 'timestamp'] = (midpoint, next_end)

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(format_subtitle_text)

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return ''.join([f"{i+1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

def gen_source_subtitles():
    """Transcription only: `src.srt` from the split sentences, without a translation."""
    from core._1_ytdlp import find_subtitle_file
    from core._5_split_sub import split_source_lines
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    with open(_3_2_SPLIT_BY_MEANING, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # The subtitles of the user keep their lines
    df_source = pd.DataFrame({'Source': lines if find_subtitle_file() else split_source_lines(lines)})
    align_timestamp(df_text, df_source, [('src.srt', ['Source'])], _OUTPUT_DIR, for_display=False)
    console.print(Panel("[bold green]🎉📝 Source subtitles are ready: `output/src.srt`[/bold green]"))

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。，、？！：；')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()
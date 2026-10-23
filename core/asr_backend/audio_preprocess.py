import math
import os
from pathlib import Path
import re
import subprocess
from typing import Dict, List, Tuple
import pandas as pd
from pydub import AudioSegment
from core.utils import load_key_or, update_key
from core.utils.models import _2_CLEANED_CHUNKS, _AUDIO_DIR, _RAW_AUDIO_FILE, resolve_task_path
from core.workspace import TaskWorkspace, get_active_workspace
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint


def audio_slice_wav(path, start=None, end=None, sample_rate=16000):
    """Decode only the requested interval to mono PCM WAV for cloud ASR."""
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in (start, end)):
        raise ValueError('Audio interval must contain finite non-negative times')
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', str(start)]
    cmd += ['-i', str(path)]
    if end is not None:
        duration = end - (start or 0)
        if duration <= 0:
            raise ValueError('Audio interval must have positive duration')
        cmd += ['-t', str(duration)]
    cmd += ['-ac', '1', '-ar', str(sample_rate), '-c:a', 'pcm_s16le', '-f', 'wav', 'pipe:1']
    return subprocess.run(cmd, check=True, capture_output=True).stdout

def _ffmpeg_has_encoder(encoder_name: str) -> bool:
    """Check if the current ffmpeg installation supports a given audio encoder."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10
        )
        return encoder_name in result.stdout
    except Exception:
        return False

# Raw extraction quality. v2.2.1 extracted 32 kHz / 128 kbps; the 3.0.0 refactor
# dropped this to 16 kHz / 32 kbps. Whisper resamples to 16 kHz anyway, but every
# other consumer inherits the extraction bandwidth: Demucs vocals, the per-line
# reference clips for voice cloning (sf_fish_tts / sf_cosyvoice2 / gpt_sovits /
# f5tts) and the timing analysis. See docs/audio-extract-quality.md.
RAW_AUDIO_SAMPLE_RATE = 32000
RAW_AUDIO_BITRATE = '128k'

def raw_audio_settings():
    """Sample rate and MP3 bitrate for raw.mp3; the keys are optional so old config.yaml files keep working."""
    sample_rate = int(load_key_or('audio.raw_sample_rate', RAW_AUDIO_SAMPLE_RATE))
    bitrate = str(load_key_or('audio.raw_bitrate', RAW_AUDIO_BITRATE))
    if sample_rate < 16000:
        raise ValueError('audio.raw_sample_rate must be at least 16000 Hz for speech recognition')
    return sample_rate, bitrate

def _raw_audio_command(input_file: str):
    sample_rate, bitrate = raw_audio_settings()
    raw_out = globals().get('resolve_task_path', lambda p: p)(_RAW_AUDIO_FILE)
    if _ffmpeg_has_encoder('libmp3lame'):
        return [
            'ffmpeg', '-y', '-i', input_file, '-vn',
            # Reconcile decoded samples with the source presentation clock.
            '-af', 'aresample=async=1:first_pts=0',
            '-c:a', 'libmp3lame', '-b:a', bitrate,
            '-ar', str(sample_rate), '-ac', '1',
            '-metadata', 'encoding=UTF-8', raw_out
        ]
    # Fallback: conda-forge ffmpeg often lacks libmp3lame.
    # Output as WAV (PCM) which all ffmpeg builds support.
    # Downstream readers (pydub, librosa, whisperX) detect format by
    # file header, not extension, so .mp3 path with WAV content works.
    rprint("[yellow]⚠️ libmp3lame not found in ffmpeg, falling back to WAV (PCM) encoding[/yellow]")
    return [
        'ffmpeg', '-y', '-i', input_file, '-vn',
        '-af', 'aresample=async=1:first_pts=0',
        '-c:a', 'pcm_s16le', '-ar', str(sample_rate), '-ac', '1',
        '-f', 'wav', raw_out
    ]

def normalize_audio_volume(audio_path, output_path, target_db = -20.0, format = "wav", peak_ceiling_db = -1.0):
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_db - audio.dBFS
    # pydub applies gain on integer samples, so a quiet track with sharp peaks
    # (speech separated by long silences) would clip. Cap the gain so peaks stay
    # under the ceiling; attenuation is never limited.
    headroom = peak_ceiling_db - audio.max_dBFS
    if not math.isfinite(change_in_dBFS) or not math.isfinite(headroom):
        change_in_dBFS = 0.0
    elif change_in_dBFS > headroom:
        rprint(f"[yellow]⚠️ Gain limited to {headroom:+.1f}dB (wanted {change_in_dBFS:+.1f}dB) to keep peaks under {peak_ceiling_db:.1f}dBFS[/yellow]")
        change_in_dBFS = headroom
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(output_path, format=format)
    rprint(f"[green]✅ Audio normalized from {audio.dBFS:.1f}dB to {audio.dBFS + change_in_dBFS:.1f}dB[/green]")
    return output_path

def convert_video_to_audio(video_file: str):
    resolve = globals().get('resolve_task_path', lambda p: p)
    audio_dir = resolve(_AUDIO_DIR)
    raw_file = resolve(_RAW_AUDIO_FILE)
    os.makedirs(audio_dir, exist_ok=True)
    if not os.path.exists(raw_file):
        rprint("[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
        subprocess.run(_raw_audio_command(video_file), check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{raw_file}> with FFmpeg\n[/green]")

def prepare_audio_for_asr(audio_file: str):
    resolve = globals().get('resolve_task_path', lambda p: p)
    audio_dir = resolve(_AUDIO_DIR)
    raw_file = resolve(_RAW_AUDIO_FILE)
    os.makedirs(audio_dir, exist_ok=True)
    if not os.path.exists(raw_file):
        rprint("[blue]🎵 Preparing uploaded audio for ASR with FFmpeg ......[/blue]")
        subprocess.run(_raw_audio_command(audio_file), check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎵 Prepared <{audio_file}> as <{raw_file}>\n[/green]")

def get_audio_duration(audio_file: str) -> float:
    """Get the duration of an audio file using ffmpeg."""
    cmd = ['ffmpeg', '-i', audio_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    output = stderr.decode('utf-8', errors='ignore')
    
    try:
        duration_str = [line for line in output.split('\n') if 'Duration' in line][0]
        duration_parts = duration_str.split('Duration: ')[1].split(',')[0].split(':')
        duration = float(duration_parts[0])*3600 + float(duration_parts[1])*60 + float(duration_parts[2])
    except Exception as e:
        print(f"[red]❌ Error: Failed to get audio duration: {e}[/red]")
        duration = 0
    return duration

def split_audio(audio_file: str, target_len: float = 30*60, win: float = 60) -> List[Tuple[float, float]]:
    ## 在 [target_len-win, target_len+win] 区间内用 pydub 检测静默，切分音频
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len + win:
        return [(0, duration)]
    segments, pos = [], 0.0
    safe_margin = 0.5  # 静默点前后安全边界，单位秒

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration))
            break

        threshold = pos + target_len
        ws, we = int((threshold - win) * 1000), int((threshold + win) * 1000)
        
        # 获取完整的静默区域
        silence_regions = detect_silence(audio[ws:we], min_silence_len=int(safe_margin*1000), silence_thresh=-30)
        silence_regions = [(s/1000 + (threshold - win), e/1000 + (threshold - win)) for s, e in silence_regions]
        # 筛选长度足够（至少1秒）且位置适合的静默区域
        valid_regions = [
            (start, end) for start, end in silence_regions 
            if (end - start) >= (safe_margin * 2) and threshold <= start + safe_margin <= threshold + win
        ]
        
        if valid_regions:
            start, end = valid_regions[0]
            split_at = start + safe_margin  # 在静默区域起始点后0.5秒处切分
        else:
            rprint(f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]")
            split_at = threshold
            
        segments.append((pos, split_at))
        pos = split_at

    rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
    return segments

def _split_segment_text_into_tokens(seg_text: str) -> list[str]:
    tokens = re.findall(
        r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7f][，。、？！：；,.?!\"'”’）)\]]*|[^\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7f]+",
        seg_text,
    )
    return tokens or [seg_text]


def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker_id, if not exists, set to None
        speaker_id = segment.get('speaker_id', None)

        words = segment.get('words')
        if not words:
            # Synthesize token-level entries with interpolated timestamps when an ASR
            # backend returns segment text without per-word timestamps.
            seg_text = (segment.get('text') or '').strip()
            if not seg_text:
                continue
            tokens = _split_segment_text_into_tokens(seg_text)
            seg_start = segment.get('start')
            seg_end = segment.get('end')
            if seg_start is None and seg_end is None:
                words = [{'word': tok} for tok in tokens]
            else:
                s_val = float(seg_start if seg_start is not None else (all_words[-1]['end'] if all_words else 0.0))
                e_val = float(seg_end if seg_end is not None else s_val)
                weights = [max(1, len(re.sub(r"[^\w\s]", "", tok))) for tok in tokens]
                total_w = sum(weights) or len(tokens)
                done = 0
                words = []
                for tok, w in zip(tokens, weights):
                    w_start = s_val + (e_val - s_val) * done / total_w
                    done += w
                    w_end = s_val + (e_val - s_val) * done / total_w
                    words.append({
                        'word': tok,
                        'start': round(w_start, 3),
                        'end': round(w_end, 3),
                    })

        for word in words:
            # Check word length
            if len(word["word"]) > 80:
                rprint(f"[yellow]⚠️ Warning: Detected word longer than 80 characters, skipping: {word['word']}[/yellow]")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            has_start = 'start' in word and word['start'] is not None
            has_end = 'end' in word and word['end'] is not None
            if not has_start or not has_end:
                if has_start and not has_end:
                    all_words.append({
                        'text': f'{word["word"]}',
                        'start': word['start'],
                        'end': word['start'],
                        'speaker_id': speaker_id,
                    })
                elif has_end and not has_start:
                    all_words.append({
                        'text': f'{word["word"]}',
                        'start': all_words[-1]['end'] if all_words else word['end'],
                        'end': word['end'],
                        'speaker_id': speaker_id,
                    })
                elif all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': all_words[-1]['end'],
                        'end': all_words[-1]['end'],
                        'speaker_id': speaker_id
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in (segment.get('words') or words) if 'start' in w and 'end' in w and w['start'] is not None and w['end'] is not None), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': next_word["start"],
                            'end': next_word["end"],
                            'speaker_id': speaker_id
                        }
                        all_words.append(word_dict)
                    else:
                        raise Exception(f"No next word with timestamp found for the current word : {word}")
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': word['start'],
                    'end': word['end'],
                    'speaker_id': speaker_id
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    cleaned_path = resolve_task_path(_2_CLEANED_CHUNKS)
    os.makedirs(os.path.dirname(cleaned_path) or 'output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = pd.DataFrame(df[df['text'].str.len() > 0]).copy()
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")
    
    # Check for and remove words longer than 80 characters
    long_words = df[df['text'].str.len() > 80]
    if not long_words.empty:
        rprint(f"[yellow]⚠️ Warning: Detected {len(long_words)} word(s) longer than 80 characters. These will be removed.[/yellow]")
        df = pd.DataFrame(df[df['text'].str.len() <= 80]).copy()

    word_records = [
        {
            "text": str(row.get("text", "")).strip('"').strip(),
            "start": float(row.get("start") or 0.0),
            "end": float(row.get("end") or 0.0),
            "speaker_id": None if bool(pd.isna(row.get("speaker_id"))) else row.get("speaker_id"),
        }
        for _, row in df.iterrows()
    ]
    ws = (
        get_active_workspace()
        if str(_2_CLEANED_CHUNKS).startswith("output/")
        else TaskWorkspace(Path(cleaned_path).parent.parent if Path(cleaned_path).parent.name == "log" else Path(cleaned_path).parent)
    )
    ws.save_words(word_records)
    
    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_excel(cleaned_path, index=False)
    rprint(f"[green]📊 Excel file saved to {cleaned_path}[/green]")

def save_language(language: str):
    update_key("whisper.detected_language", language)

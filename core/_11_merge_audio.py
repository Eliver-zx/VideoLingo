import os
import pandas as pd
from pydub import AudioSegment
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from core.utils import check_cancel, rprint
from core.utils.models import _8_1_AUDIO_TASK, _AUDIO_SEGS_DIR, _DUB_WAV_FILE, resolve_task_path
from core.utils.task_literals import parse_task_literal
console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'
DUB_WAV_FILE = _DUB_WAV_FILE
DUB_SAMPLE_RATE = 32000

DUB_SUB_FILE = 'output/dub.srt'
DUB_SRC_SUB_FILE = 'output/dub_src.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"

def load_and_flatten_data(excel_file):
    """Load and flatten Excel data"""
    df = pd.read_excel(excel_file)
    lines = [parse_task_literal(line) for line in df['lines'].tolist()]
    lines = [item for sublist in lines for item in sublist]
    
    new_sub_times = [parse_task_literal(time) for time in df['new_sub_times'].tolist()]
    new_sub_times = [item for sublist in new_sub_times for item in sublist]
    
    return df, lines, new_sub_times

def get_audio_files(df):
    """Generate a list of audio file paths"""
    _resolve = globals().get("resolve_task_path", lambda p: p)
    audios = []
    for index, row in df.iterrows():
        number = row['number']
        line_count = len(parse_task_literal(row['lines']))
        for line_index in range(line_count):
            temp_file = _resolve(OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}"))
            audios.append(temp_file)
    return audios

def process_audio_segment(audio_file, sample_rate=DUB_SAMPLE_RATE):
    """Load a single audio segment directly as lossless mono PCM without lossy MP3 transcoding."""
    audio_segment = AudioSegment.from_file(audio_file)
    return audio_segment.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)

def merge_audio_segments(audios, new_sub_times, sample_rate=DUB_SAMPLE_RATE):
    merged_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn()) as progress:
        merge_task = progress.add_task("🎵 Merging audio segments...", total=len(audios))
        
        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            check_cancel()
            if not os.path.exists(audio_file):
                console.print(f"[bold yellow]⚠️  Warning: File {audio_file} does not exist, skipping...[/bold yellow]")
                progress.advance(merge_task)
                continue
                
            audio_segment = process_audio_segment(audio_file, sample_rate=sample_rate)
            start_time, end_time = time_range
            
            # Anchor silence duration to actual accumulated audio length so timeline never drifts
            current_pos_sec = len(merged_audio) / 1000.0
            silence_duration = max(0.0, float(start_time) - current_pos_sec)
            silence_ms = int(round(silence_duration * 1000))
            if silence_ms > 0:
                merged_audio += AudioSegment.silent(duration=silence_ms, frame_rate=sample_rate)
                
            merged_audio += audio_segment
            progress.advance(merge_task)
    
    return merged_audio

def write_srt(path, times, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(times, lines), 1):
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time*1000)%1000):03d}"
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time*1000)%1000):03d}"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")
    rprint(f"[bold green]✅ Subtitle file created: {path}[/bold green]")

def create_srt_subtitle():
    df, lines, new_sub_times = load_and_flatten_data(resolve_task_path(_8_1_AUDIO_TASK))
    dub_sub_path = resolve_task_path(DUB_SUB_FILE)
    dub_src_sub_path = resolve_task_path(DUB_SRC_SUB_FILE)
    write_srt(dub_sub_path, new_sub_times, lines)

    # The source lines on the timeline of the dub, for bilingual subtitles in the dubbed video
    if os.path.exists(dub_src_sub_path):
        os.remove(dub_src_sub_path)
    if 'src_lines' in df.columns:
        try:
            src_lines = [line for row in df['src_lines'].tolist() for line in parse_task_literal(row)]
        except (TypeError, ValueError, SyntaxError):
            src_lines = []
        if len(src_lines) == len(lines):
            write_srt(dub_src_sub_path, new_sub_times, src_lines)

def merge_full_audio():
    """Main function: Process the complete audio merging process"""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")
    
    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(resolve_task_path(_8_1_AUDIO_TASK))
    console.print("[bold green]✅ Data loaded successfully[/bold green]")
    
    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")
    
    with console.status("[bold cyan]📝 Generating subtitle file...[/bold cyan]"):
        create_srt_subtitle()
    
    if not audios or not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0] if audios else 'None'} does not exist![/bold red]")
        return
    
    sample_rate = DUB_SAMPLE_RATE
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate)
    
    dub_wav_path = resolve_task_path(DUB_WAV_FILE)
    dub_vocal_path = resolve_task_path(DUB_VOCAL_FILE)
    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        os.makedirs(os.path.dirname(dub_wav_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(dub_vocal_path) or ".", exist_ok=True)
        merged_audio.export(dub_wav_path, format="wav")
        merged_audio.export(dub_vocal_path, format="mp3", parameters=["-b:a", "192k"])
    console.print("[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output files: {dub_wav_path}, {dub_vocal_path}[/bold green]")

if __name__ == "__main__":
    merge_full_audio()

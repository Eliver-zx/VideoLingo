from pathlib import Path
import sys
from core.utils import load_key
import subprocess

# Available voices can be listed using edge-tts --list-voices command
# Common English voices:
# en-US-JennyNeural - Female
# en-US-GuyNeural - Male  
# en-GB-SoniaNeural - Female British
# Common Chinese voices:
# zh-CN-XiaoxiaoNeural - Female
# zh-CN-YunxiNeural - Male
# zh-CN-XiaoyiNeural - Female
def edge_tts(text, save_path):
    # Load settings from config file
    edge_set = load_key("edge_tts")
    voice = edge_set.get("voice", "en-US-JennyNeural")
    
    # Create output directory if it doesn't exist
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voice, f"--text={text}", "--write-media", str(speech_file_path)]
    subprocess.run(cmd, check=True)
    # edge-tts outputs MP3 bytes even when --write-media ends in .wav; convert to RIFF PCM WAV so wave/soundfile can read it
    if speech_file_path.is_file() and speech_file_path.suffix.lower() == ".wav":
        if speech_file_path.read_bytes()[:4] != b"RIFF":
            from pydub import AudioSegment
            AudioSegment.from_file(speech_file_path).export(speech_file_path, format="wav")
    print(f"Audio saved to {speech_file_path}")

if __name__ == "__main__":
    edge_tts("Today is a good day!", "edge_tts.wav")

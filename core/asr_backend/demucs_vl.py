import gc
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from rich import print as rprint
from rich.console import Console
from torch.cuda import is_available as is_cuda_available

from core.asr_backend.audio_preprocess import _ffmpeg_has_encoder
from core.utils.config_utils import load_key_or
from core.utils.models import *

try:
    from demucs.api import Separator as _DemucsSeparator  # type: ignore[import-not-found]
    from demucs.apply import BagOfModels  # type: ignore[import-not-found]
    from demucs.audio import AudioFile, convert_audio_channels  # type: ignore[import-not-found]
    from demucs.pretrained import get_model  # type: ignore[import-not-found]
except ImportError:
    class _DemucsSeparator:
        def update_parameter(self, **kwargs: Any) -> None:
            pass

    BagOfModels = Any  # type: ignore[assignment,misc]
    AudioFile = None  # type: ignore[assignment]
    get_model = None  # type: ignore[assignment]

    def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        src_channels = wav.shape[-2]
        if src_channels == channels:
            return wav
        if channels == 1:
            return wav.mean(dim=-2, keepdim=True)
        if src_channels == 1:
            return wav.expand(*wav.shape[:-2], channels, wav.shape[-1]).contiguous()
        if src_channels > channels:
            return wav[..., :channels, :]
        raise ValueError(f"Cannot convert audio from {src_channels} to {channels} channels")


VOCAL_SEPARATION_MODELS = {
    "htdemucs": {
        "backend": "demucs",
        "model": "htdemucs",
        "label": "HTDemucs",
    },
    "mdx": {
        "backend": "audio_separator",
        "model": "Kim_Vocal_2.onnx",
        "label": "MDX-Net (Kim_Vocal_2)",
    },
    "scnet": {
        "backend": "torch",
        "model": "SCNet-large_starrytong_fixed.ckpt",
        "label": "SCNet-Large",
    },
    "mel_band_roformer": {
        "backend": "audio_separator",
        "model": "vocals_mel_band_roformer.ckpt",
        "label": "Mel-Band-RoFormer",
    },
}


def _config_or(key: str, default: str) -> str:
    try:
        val = load_key_or(key, default)
        return str(val) if val else default
    except (FileNotFoundError, KeyError):
        return default


def resolve_separation_model(requested: Optional[str] = None) -> str:
    raw = requested if requested is not None else _config_or("vocal_separation_model", "htdemucs")
    key = str(raw or "htdemucs").strip().lower()
    if key not in VOCAL_SEPARATION_MODELS:
        valid = ", ".join(VOCAL_SEPARATION_MODELS)
        raise ValueError(f"vocal_separation_model must be one of {valid}, got {raw!r}")
    return key


def separation_backend_package(model_key: Optional[str] = None) -> str:
    resolved = resolve_separation_model(model_key)
    return VOCAL_SEPARATION_MODELS[resolved]["backend"]


class PreloadedSeparator(_DemucsSeparator):  # type: ignore[misc]
    def __init__(self, model: Any, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)


def _match_stem_shape(stem: torch.Tensor, target_channels: int, target_samples: int) -> torch.Tensor:
    stem = convert_audio_channels(stem, target_channels)
    cur_samples = stem.shape[-1]
    if cur_samples > target_samples:
        return stem[..., :target_samples]
    if cur_samples < target_samples:
        pad = torch.zeros((*stem.shape[:-1], target_samples - cur_samples), dtype=stem.dtype, device=stem.device)
        return torch.cat([stem, pad], dim=-1)
    return stem


def _classify_stem_filename(filename: str) -> Optional[str]:
    """Identify whether an audio-separator output file is 'vocals' or 'instrumental'.

    Matches the parenthesized stem tag `_(Vocals)_` / `_(Instrumental)_` first so model
    filenames containing 'Vocal' (e.g. `Kim_Vocal_2`, `vocals_mel_band_roformer`) never
    cause the instrumental file to be misclassified as vocals.
    """
    tags = [m.lower().strip() for m in re.findall(r"\(([^()]+)\)", filename)]
    for tag in reversed(tags):
        if tag in ("vocals", "vocal", "lead vocals", "primary stem"):
            return "vocals"
        if tag in ("instrumental", "inst", "other", "no_vocals", "no_vocal", "karaoke", "secondary stem"):
            return "instrumental"
    stem_base = Path(filename).stem.lower()
    if stem_base in ("vocals", "vocal", "chunk_vocals"):
        return "vocals"
    if stem_base in ("instrumental", "inst", "other", "no_vocals", "chunk_instrumental"):
        return "instrumental"
    return None


class AudioSeparatorAdapter:
    """Wrap python-audio-separator with the same separate_tensor contract as Demucs."""

    samplerate = 44100
    audio_channels = 2

    def __init__(self, model_filename: str, model_dir: Optional[str] = None):
        from audio_separator.separator import Separator as FileSeparator  # type: ignore[import-not-found]

        cache_dir = model_dir or _config_or("model_dir", "./_model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.model_filename = model_filename
        self.separator = FileSeparator(
            model_file_dir=cache_dir,
            output_format="WAV",
            sample_rate=self.samplerate,
        )
        self.separator.load_model(model_filename=model_filename)

    def separate_tensor(self, wav: torch.Tensor, sr: Optional[int] = None):
        sample_rate = sr or self.samplerate
        channels, samples = wav.shape
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_wav = tmp_path / "chunk_in.wav"
            sf.write(str(input_wav), wav.detach().cpu().t().numpy(), sample_rate, subtype="FLOAT")

            # Sync output_dir on both Separator and its loaded architecture instance
            self.separator.output_dir = str(tmp_path)
            model_instance = getattr(self.separator, "model_instance", None)
            if model_instance is not None:
                model_instance.output_dir = str(tmp_path)

            out_files = self.separator.separate(str(input_wav))

            vocal_path = None
            inst_path = None
            for f in out_files:
                candidate = Path(f)
                if not candidate.is_absolute():
                    in_tmp = tmp_path / candidate
                    candidate = in_tmp if in_tmp.exists() else Path(os.getcwd()) / candidate
                role = _classify_stem_filename(candidate.name)
                if role == "vocals":
                    vocal_path = candidate
                elif role == "instrumental":
                    inst_path = candidate

            if vocal_path is None or not vocal_path.exists():
                raise RuntimeError(f"audio-separator did not produce a vocal stem from {self.model_filename}: {out_files}")

            v_data, _ = sf.read(str(vocal_path), dtype="float32", always_2d=True)
            vocals = _match_stem_shape(torch.from_numpy(v_data.T), channels, samples)

            if inst_path is not None and inst_path.exists():
                i_data, _ = sf.read(str(inst_path), dtype="float32", always_2d=True)
                background = _match_stem_shape(torch.from_numpy(i_data.T), channels, samples)
            else:
                background = wav.detach().cpu() - vocals

        return wav, {"vocals": vocals, "other": background}


# Separated at once. Memory use follows this, not the length of the video.
CHUNK_SECONDS = 300
# Audio on both sides of a chunk that the model hears and whose output is dropped
CONTEXT_SECONDS = 5

class StemWriter:
    """Encode a stem as MP3 through FFmpeg's libmp3lame while it is being separated.

    demucs.audio.save_audio encodes with lameenc, which writes no LAME/Xing header, so
    decoders cannot trim the encoder delay and the stem plays ~25 ms late. FFmpeg's
    header lets FFmpeg, pydub and soundfile decode it sample-aligned with raw.mp3.
    Without libmp3lame, fall back to PCM WAV content under the same name (as raw.mp3 does).
    """
    def __init__(self, path, samplerate, channels, bitrate="128k"):
        if _ffmpeg_has_encoder("libmp3lame"):
            codec = ["-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3"]
        else:
            codec = ["-c:a", "pcm_s16le", "-f", "wav"]
        self.process = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(samplerate), "-ac", str(channels),
             "-i", "pipe:0", *codec, str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, wav):
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(wav.clamp(-1, 1).t().contiguous().numpy().astype("<f4").tobytes())
        except (BrokenPipeError, OSError):
            self.close()
            raise

    def close(self):
        _, stderr = self.process.communicate()
        if self.process.returncode:
            raise RuntimeError(f"FFmpeg could not write the stem: {stderr.decode('utf-8', errors='replace').strip()}")


def _probe_audio_channels(audio_file) -> int:
    if AudioFile is not None:
        return AudioFile(audio_file).channels(0)
    try:
        return sf.info(str(audio_file)).channels
    except Exception:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)],
            check=True, capture_output=True, text=True,
        )
        return int(result.stdout.strip())


def decode_pcm(audio_file, pcm_file, samplerate):
    """Decode to raw float32 and return the number of channels.

    FFmpeg trims the MP3 encoder delay of raw.mp3. Demucs' separate_audio_file() reads
    with sphn first, which keeps that delay and made the stems ~35 ms late relative to
    raw.mp3 (32 kHz LAME priming).
    """
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(audio_file), "-map", "0:a:0", "-threads", "1",
                    "-f", "f32le", "-ar", str(samplerate), str(pcm_file)], check=True, capture_output=True)
    return _probe_audio_channels(audio_file)

def separate_in_chunks(separator, pcm_file, pcm_channels, samplerate, channels, vocal_writer, background_writer):
    frame_bytes = 4 * pcm_channels
    total = os.path.getsize(pcm_file) // frame_bytes
    chunk, context = CHUNK_SECONDS * samplerate, CONTEXT_SECONDS * samplerate
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        left, right = max(start - context, 0), min(end + context, total)
        if total > chunk:
            rprint(f"🎵 Separating {start // samplerate}s - {end // samplerate}s of {total // samplerate}s...")
        frames = np.fromfile(pcm_file, dtype="<f4", count=(right - left) * pcm_channels, offset=left * frame_bytes)
        wav = convert_audio_channels(torch.from_numpy(frames.astype(np.float32)).view(-1, pcm_channels).t(), channels)
        _, outputs = separator.separate_tensor(wav, samplerate)
        keep = slice(start - left, end - left)
        other_stems = [audio for source, audio in outputs.items() if source != 'vocals']
        background = other_stems[0].clone()
        for stem in other_stems[1:]:
            background = background + stem
        vocal_writer.write(outputs['vocals'][..., keep].cpu())
        background_writer.write(background[..., keep].cpu())
        del outputs, background, wav, frames
        gc.collect()

def demucs_audio(model_key: Optional[str] = None):
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip vocal separation.[/yellow]")
        return

    resolved = resolve_separation_model(model_key)
    spec = VOCAL_SEPARATION_MODELS[resolved]
    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    console.print(f"🤖 Loading <{spec['model']}> vocal separation model...")
    if spec["backend"] == "demucs":
        if get_model is None:
            raise RuntimeError("Demucs is not installed. Install demucs or choose mdx/scnet/mel_band_roformer.")
        model = get_model(spec["model"])
        separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)
        samplerate, audio_channels = model.samplerate, model.audio_channels
    elif spec["backend"] == "torch":
        from core.asr_backend.scnet_vl import SCNetSeparatorAdapter
        model = None
        separator = SCNetSeparatorAdapter(
            model_filename=spec["model"],
            model_dir=_config_or("model_dir", "./_model_cache"),
        )
        samplerate, audio_channels = separator.samplerate, separator.audio_channels
    else:
        model = None
        separator = AudioSeparatorAdapter(model_filename=spec["model"])
        samplerate, audio_channels = separator.samplerate, separator.audio_channels

    console.print("🎵 Separating audio...")
    # The stems are written next to their final place and moved there when they are
    # complete, so an interrupted run does not leave half a track behind.
    with tempfile.TemporaryDirectory(dir=_AUDIO_DIR) as tmp:
        pcm_file = Path(tmp) / "raw.f32le"
        vocal_file, background_file = Path(tmp) / "vocal.mp3", Path(tmp) / "background.mp3"
        pcm_channels = decode_pcm(_RAW_AUDIO_FILE, pcm_file, samplerate)
        vocal_writer = StemWriter(vocal_file, samplerate, audio_channels)
        background_writer = StemWriter(background_file, samplerate, audio_channels)
        try:
            separate_in_chunks(separator, pcm_file, pcm_channels, samplerate, audio_channels,
                               vocal_writer, background_writer)
        finally:
            for writer in (vocal_writer, background_writer):
                if writer.process.poll() is None:
                    writer.close()
        for writer in (vocal_writer, background_writer):
            writer.close()
        os.replace(vocal_file, _VOCAL_AUDIO_FILE)
        os.replace(background_file, _BACKGROUND_AUDIO_FILE)

    # Clean up memory
    del model, separator
    gc.collect()

    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_audio()

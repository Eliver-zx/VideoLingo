"""Offline regressions for raw extraction quality and reference audio (#533); no models, TTS or network."""

import ast
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from core.asr_backend import audio_preprocess
from core.tts_backend import sf_fishtts
from core.utils import config_utils

ROOT = Path(__file__).resolve().parents[1]


def configure_extraction(
    monkeypatch, directory, config, encoder=True, real_encoder_probe=False
):
    monkeypatch.setattr(audio_preprocess, "_AUDIO_DIR", str(directory))
    monkeypatch.setattr(audio_preprocess, "_RAW_AUDIO_FILE", str(directory / "raw.mp3"))
    monkeypatch.setattr(audio_preprocess, "resolve_task_path", lambda p: p)
    monkeypatch.setattr(
        audio_preprocess,
        "load_key_or",
        lambda key, default: config.get(key, default),
    )
    if not real_encoder_probe:
        monkeypatch.setattr(audio_preprocess, "_ffmpeg_has_encoder", lambda _: encoder)


# ---------------------------------------------------------------------------
# Step 2: raw extraction quality (v2.2.1 parity, configurable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoder", [True, False], ids=["mp3", "pcm-fallback"])
@pytest.mark.parametrize(
    "config,rate,bitrate",
    [
        ({}, "32000", "128k"),
        (
            {"audio.raw_sample_rate": 44100, "audio.raw_bitrate": "192k"},
            "44100",
            "192k",
        ),
    ],
    ids=["old-config-without-keys", "configured"],
)
def test_raw_extraction_uses_v2_quality_by_default(
    tmp_path, monkeypatch, encoder, config, rate, bitrate
):
    configure_extraction(monkeypatch, tmp_path, config, encoder)
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    audio_preprocess.convert_video_to_audio("video.mp4")
    audio_preprocess.prepare_audio_for_asr("upload.wav")
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("-ar") + 1] == rate
        assert command[command.index("-af") + 1] == "aresample=async=1:first_pts=0"
        if encoder:
            assert command[command.index("-b:a") + 1] == bitrate
        else:
            assert (
                "-b:a" not in command
                and command[command.index("-c:a") + 1] == "pcm_s16le"
            )


def test_sample_rate_below_recognition_rate_is_rejected(tmp_path, monkeypatch):
    configure_extraction(monkeypatch, tmp_path, {"audio.raw_sample_rate": 8000})
    with pytest.raises(ValueError, match="audio.raw_sample_rate"):
        audio_preprocess.raw_audio_settings(high_quality=True)
    configure_extraction(monkeypatch, tmp_path, {"audio.asr_sample_rate": 8000})
    with pytest.raises(ValueError, match="audio.asr_sample_rate"):
        audio_preprocess.raw_audio_settings(high_quality=False)


def test_asr_lightweight_vs_dubbing_high_quality_audio_modes(tmp_path, monkeypatch):
    """Pure ASR / subtitle mode uses 16 kHz mono 48k MP3, while dubbing / demucs uses stereo high-quality MP3."""
    from core import _2_asr, _9_refer_audio
    from core.workspace import TaskWorkspace, workspace_context

    configure_extraction(
        monkeypatch,
        tmp_path,
        {"audio.raw_sample_rate": 44100, "audio.raw_bitrate": "192k"},
        encoder=True,
    )
    ws = TaskWorkspace(tmp_path / "ws_modes")
    ws.ensure_dirs()
    video_path = tmp_path / "lecture.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    ws.init_from_local_file(video_path, title="Lecture")
    ws.update_manifest(
        execution={"desired_stage": "subtitles", "dubbing_enabled": False}
    )

    commands: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: commands.append(command)
    )

    with workspace_context(ws):
        # 1. Subtitles-only mode without demucs -> lightweight 16 kHz mono 48k
        _2_asr.prepare_audio(str(video_path), "video", demucs=False)
        assert len(commands) == 1
        cmd_asr = commands[-1]
        assert cmd_asr[cmd_asr.index("-ar") + 1] == "16000"
        assert cmd_asr[cmd_asr.index("-ac") + 1] == "1"
        assert cmd_asr[cmd_asr.index("-b:a") + 1] == "48k"

        # 2. Dubbing enabled -> high-quality 44.1 kHz stereo 192k
        ws.update_manifest(execution={"desired_stage": "all", "dubbing_enabled": True})
        _2_asr.prepare_audio(str(video_path), "video", demucs=False)
        assert len(commands) == 2
        cmd_dub = commands[-1]
        assert cmd_dub[cmd_dub.index("-ar") + 1] == "44100"
        assert cmd_dub[cmd_dub.index("-ac") + 1] == "2"
        assert cmd_dub[cmd_dub.index("-b:a") + 1] == "192k"

        # 3. _ensure_high_quality_raw_audio upgrades a 16 kHz mono raw.mp3 and removes stale vocal/background stems
        fake_raw = tmp_path / "raw.mp3"
        fake_raw.write_bytes(b"fake")
        stale_vocal = ws.audio_dir / "vocal.mp3"
        stale_bg = ws.audio_dir / "background.mp3"
        stale_vocal.write_bytes(b"stale-vocal")
        stale_bg.write_bytes(b"stale-bg")
        monkeypatch.setattr(
            _9_refer_audio.sf,
            "info",
            lambda _: type("Info", (), {"samplerate": 16000, "channels": 1})(),
        )
        _9_refer_audio._ensure_high_quality_raw_audio(str(fake_raw))
        assert len(commands) == 3
        cmd_upg = commands[-1]
        assert cmd_upg[cmd_upg.index("-ar") + 1] == "44100"
        assert cmd_upg[cmd_upg.index("-ac") + 1] == "2"
        assert cmd_upg[cmd_upg.index("-b:a") + 1] == "192k"
        assert not stale_vocal.exists()
        assert not stale_bg.exists()


def test_ffmpeg_has_encoder_raises_runtime_error_when_ffmpeg_missing(
    tmp_path, monkeypatch
):
    configure_extraction(monkeypatch, tmp_path, {}, real_encoder_probe=True)
    monkeypatch.setattr(
        audio_preprocess._ffmpeg_has_encoder, "_cache", {}, raising=False
    )

    def raise_missing(*args, **kwargs):
        raise FileNotFoundError(3, "系统找不到指定的路径。")

    monkeypatch.setattr(subprocess, "run", raise_missing)
    with pytest.raises(RuntimeError, match="FFmpeg executable not found in PATH"):
        audio_preprocess._ffmpeg_has_encoder("libmp3lame")


def test_audio_preprocess_imports_resolve_task_path():
    source = (ROOT / "core/asr_backend/audio_preprocess.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "core.utils.models"
        for alias in node.names
    }
    assert "resolve_task_path" in imported_names


def test_windows_ffmpeg_path_sanitizes_broken_dirs_and_hooks_popen(
    tmp_path, monkeypatch
):
    from core.utils import models

    fake_bin_dir = tmp_path / "ffmpeg_bin"
    fake_bin_dir.mkdir()
    fake_ffmpeg = fake_bin_dir / "ffmpeg.exe"
    fake_ffmpeg.write_bytes(b"MZ")
    broken_dir = tmp_path / "nonexistent_dir_causing_winerror_3"

    monkeypatch.setattr(models.os, "name", "nt")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv(
        "PATH",
        f'"{broken_dir}"{os.pathsep}"{fake_bin_dir}"',
    )
    monkeypatch.setattr(models.shutil, "which", lambda name: str(fake_ffmpeg))

    captured_kwargs = {}

    def fake_orig_init(self, args, *p_args, **p_kwargs):
        captured_kwargs["args"] = args
        captured_kwargs["executable"] = p_kwargs.get("executable")

    monkeypatch.setattr(models.subprocess.Popen, "__init__", fake_orig_init)
    models.ensure_windows_ffmpeg_path()

    cleaned_path_entries = os.environ["PATH"].split(os.pathsep)
    assert str(broken_dir) not in cleaned_path_entries
    assert any(
        os.path.realpath(p) == os.path.realpath(str(fake_bin_dir))
        for p in cleaned_path_entries
        if os.path.isdir(p)
    )

    dummy = object.__new__(models.subprocess.Popen)
    models.subprocess.Popen.__init__(dummy, ["ffmpeg", "-nostdin", "-encoders"])
    assert captured_kwargs["args"] == ["ffmpeg", "-nostdin", "-encoders"]
    assert os.path.realpath(captured_kwargs["executable"]) == os.path.realpath(
        str(fake_ffmpeg)
    )


def band_level(pcm_path, low, high):
    import numpy as np

    pcm = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(pcm_path),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    samples = np.frombuffer(pcm, dtype=np.float32)
    n = 8192
    frames = samples[: len(samples) // n * n].reshape(-1, n) * np.hanning(n)
    power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / 48000)
    return 10 * math.log10(
        float(np.mean(power[(freqs >= low) & (freqs < high)])) + 1e-20
    )


def test_extracted_audio_keeps_high_band(tmp_path, monkeypatch):
    """White noise keeps its 9-14 kHz energy under the default settings; the 3.0.0 16 kHz path lost it."""
    if not shutil.which("ffmpeg"):
        pytest.skip("Real extraction test requires FFmpeg")
    pytest.importorskip("numpy")
    source = tmp_path / "noise.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=white:sample_rate=48000:duration=3:amplitude=0.3:seed=7",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    configure_extraction(monkeypatch, tmp_path, {}, real_encoder_probe=True)
    audio_preprocess.convert_video_to_audio(str(source))
    restored = tmp_path / "raw.mp3"
    legacy = tmp_path / "legacy.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(legacy),
        ],
        check=True,
        capture_output=True,
    )
    for path, expected_drop in ((restored, 6.0), (legacy, 40.0)):
        drop = band_level(path, 1000, 3000) - band_level(path, 9000, 14000)
        if expected_drop < 10:
            assert drop < expected_drop, (
                f"{path.name} lost its high band ({drop:.1f} dB)"
            )
        else:
            assert drop > expected_drop, (
                f"{path.name} unexpectedly kept its high band ({drop:.1f} dB)"
            )


# ---------------------------------------------------------------------------
# Normalization: gain must never push peaks into clipping
# ---------------------------------------------------------------------------


def test_normalize_limits_gain_to_peak_headroom(tmp_path, monkeypatch):
    from pydub import AudioSegment
    from pydub.generators import Sine

    burst = Sine(440, sample_rate=16000).to_audio_segment(duration=60, volume=-1.0)
    sparse = burst + AudioSegment.silent(duration=10000, frame_rate=16000)
    source = tmp_path / "sparse.wav"
    sparse.export(source, format="wav")
    assert -20.0 - sparse.dBFS > -1.0 - sparse.max_dBFS, (
        "fixture must want more gain than the peak allows"
    )
    output = audio_preprocess.normalize_audio_volume(
        str(source), str(tmp_path / "normalized.wav")
    )
    result = AudioSegment.from_wav(output)
    assert -1.3 <= result.max_dBFS <= -0.9


def test_normalize_still_attenuates_loud_input(tmp_path, monkeypatch):
    from pydub import AudioSegment
    from pydub.generators import Sine

    loud = Sine(440, sample_rate=16000).to_audio_segment(duration=1000, volume=-1.0)
    source = tmp_path / "loud.wav"
    loud.export(source, format="wav")
    result = AudioSegment.from_wav(
        audio_preprocess.normalize_audio_volume(
            str(source), str(tmp_path / "normalized.wav")
        )
    )
    assert abs(result.dBFS - (-20.0)) < 0.2


def test_normalize_survives_silence(tmp_path, monkeypatch):
    from pydub import AudioSegment

    source = tmp_path / "silence.wav"
    AudioSegment.silent(duration=500, frame_rate=16000).export(source, format="wav")
    output = audio_preprocess.normalize_audio_volume(
        str(source), str(tmp_path / "normalized.wav")
    )
    assert Path(output).stat().st_size > 0


# ---------------------------------------------------------------------------
# SiliconFlow Fish custom voice: merged reference keeps 44.1 kHz
# ---------------------------------------------------------------------------


def test_custom_voice_reference_is_merged_at_44k(tmp_path, monkeypatch):
    from pydub import AudioSegment
    from pydub.generators import Sine

    clips = []
    for index in (1, 2):
        clip = tmp_path / f"{index}.wav"
        Sine(300 * index, sample_rate=44100).to_audio_segment(duration=300).export(
            clip, format="wav"
        )
        clips.append(str(clip))
    output = tmp_path / "combined_reference.wav"
    assert sf_fishtts.merge_audio(clips, str(output)) is True
    merged = AudioSegment.from_wav(output)
    assert merged.frame_rate == 44100 and merged.channels == 1
    assert abs(len(merged) - (300 + 100) * 2) <= 5


# ---------------------------------------------------------------------------
# Config: optional keys and shipped defaults
# ---------------------------------------------------------------------------


def test_optional_config_key_falls_back_without_error(monkeypatch):
    def fake_load_key(key):
        if key == "present":
            return "value"
        raise KeyError(key)

    monkeypatch.setattr(config_utils, "load_key", fake_load_key)
    assert config_utils.load_key_or("present", "default") == "value"
    assert config_utils.load_key_or("missing.key", "default") == "default"


def test_shipped_config_restores_v2_extraction_defaults():
    yaml = pytest.importorskip("yaml")
    cfg_src = (
        ROOT / "config.example.yaml"
        if (ROOT / "config.example.yaml").is_file()
        else ROOT / "config.yaml"
    )
    config = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))
    assert config["audio"] == {"raw_sample_rate": 44100, "raw_bitrate": "192k"}


def test_codebase_has_no_dynamic_globals_escape_hatches_or_swallowed_subprocess_fnf():
    """Enforce static AST guardrails across api.py, core/, and tests/."""
    py_files = [ROOT / "api.py", *sorted((ROOT / "core").rglob("*.py"))]
    globals_get_hits: list[str] = []
    unhandled_subproc_hits: list[str] = []
    hardcoded_output_open_hits: list[str] = []
    partial_exec_test_hits: list[str] = []

    for path in py_files:
        rel = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "globals"
            ):
                arg0 = ast.unparse(node.args[0]) if node.args else ""
                globals_get_hits.append(f"{rel}:{node.lineno} -> globals().get({arg0})")

            if isinstance(node, ast.Try):
                has_subproc = any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "subprocess"
                    and sub.func.attr in {"run", "Popen", "check_output", "check_call"}
                    for stmt in node.body
                    for sub in ast.walk(stmt)
                )
                if has_subproc:
                    handler_types = [
                        ast.unparse(h.type) if h.type else "<bare except>"
                        for h in node.handlers
                    ]
                    has_broad = any(
                        t in {"Exception", "BaseException", "<bare except>"}
                        for t in handler_types
                    )
                    has_fnf = any("FileNotFoundError" in t for t in handler_types)
                    if has_broad and not has_fnf:
                        unhandled_subproc_hits.append(
                            f"{rel}:{node.lineno} -> subprocess in broad except {handler_types}"
                        )

            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("output/")
            ):
                hardcoded_output_open_hits.append(
                    f"{rel}:{node.lineno} -> open({node.args[0].value!r})"
                )

    for test_path in sorted((ROOT / "tests").rglob("*.py")):
        rel = test_path.relative_to(ROOT)
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(rel))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "exec"
            ):
                partial_exec_test_hits.append(f"{rel}:{node.lineno} -> exec(...)")

    assert not globals_get_hits, f"Forbidden globals().get() found: {globals_get_hits}"
    assert not unhandled_subproc_hits, (
        f"Subprocess inside broad except without FileNotFoundError guard: {unhandled_subproc_hits}"
    )
    assert not hardcoded_output_open_hits, (
        f"Hardcoded open('output/...') without resolve_task_path: {hardcoded_output_open_hits}"
    )
    assert not partial_exec_test_hits, (
        f"Forbidden partial-module exec(...) loader in tests/: {partial_exec_test_hits}"
    )

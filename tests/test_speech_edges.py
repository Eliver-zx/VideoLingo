import numpy as np
import pytest
import sys
from types import SimpleNamespace

from core.asr_backend import speech_edges


def test_edges_keep_padding_and_internal_pauses(monkeypatch):
    monkeypatch.setattr(
        speech_edges,
        "_speech_regions",
        lambda *_: [
            {"start": 500, "end": 900},
            {"start": 1500, "end": 2000},
        ],
    )
    # The six-second internal pause is retained, not concatenated away.
    assert speech_edges.trim_speech_windows(np.zeros(3000), [(0, 3000)], 100) == [
        (400, 2100)
    ]


def test_short_opening_before_long_pause_is_retained(monkeypatch):
    monkeypatch.setattr(
        speech_edges,
        "_speech_regions",
        lambda *_: [
            {"start": 60, "end": 80},
            {"start": 400, "end": 1000},
        ],
    )
    assert speech_edges.trim_speech_windows(np.zeros(1200), [(0, 1200)], 100) == [
        (0, 1100)
    ]


def test_detection_missing_quiet_speech_keeps_original_window(monkeypatch):
    monkeypatch.setattr(speech_edges, "_speech_regions", lambda *_: [])
    assert speech_edges.trim_speech_windows(
        np.zeros(2000), [(100, 1000), (1000, 2000)], 100
    ) == [(100, 1000), (1000, 2000)]


def test_padding_cannot_cross_existing_window_boundaries(monkeypatch):
    monkeypatch.setattr(
        speech_edges,
        "_speech_regions",
        lambda *_: [
            {"start": 900, "end": 1150},
            {"start": 1900, "end": 1950},
        ],
    )
    assert speech_edges.trim_speech_windows(
        np.zeros(2000), [(0, 1000), (1000, 2000)], 100
    ) == [(800, 1000), (1000, 2000)]


def test_empty_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(
        speech_edges, "_speech_regions", lambda *_: pytest.fail("VAD should not run")
    )
    assert speech_edges.trim_speech_windows(np.zeros(0), [(0, 0)]) == [(0, 0)]
    assert speech_edges.trim_speech_windows(np.zeros(10), []) == []


@pytest.mark.parametrize("fail", [False, True])
def test_vad_restores_torch_threads_even_on_failure(monkeypatch, fail):
    restored = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            get_num_threads=lambda: 8,
            set_num_threads=restored.append,
            from_numpy=lambda a: a,
        ),
    )

    def detect(audio, model, **kwargs):
        if fail:
            raise RuntimeError("inference failed")
        kwargs["progress_tracking_callback"](100)
        return [{"start": 0, "end": len(audio)}]

    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(load_silero_vad=lambda: object(), get_speech_timestamps=detect),
    )
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            speech_edges._speech_regions(np.zeros(10), 16000)
    else:
        assert speech_edges._speech_regions(np.zeros(10), 16000) == [
            {"start": 0, "end": 10}
        ]
    assert restored == [1, 8]


def test_unicode_safe_jit_load_streams_non_ascii_paths(tmp_path):
    """torch.jit.load receives an open binary stream rather than a raw str/Path so Windows Chinese paths work."""
    cn_dir = tmp_path / "中文虚拟环境" / "silero_vad" / "data"
    cn_dir.mkdir(parents=True)
    jit_file = cn_dir / "silero_vad.jit"
    jit_file.write_bytes(b"FAKE_JIT_ARCHIVE")

    received = []

    def fake_orig_load(f, *args, **kwargs):
        received.append(
            (
                type(f).__name__,
                hasattr(f, "read"),
                f.read() if hasattr(f, "read") else None,
            )
        )
        return "LOADED_MODEL"

    fake_torch = SimpleNamespace(jit=SimpleNamespace(load=fake_orig_load))
    with speech_edges._unicode_safe_jit_load(fake_torch):
        res = fake_torch.jit.load(str(jit_file), map_location="cpu")
    assert res == "LOADED_MODEL"
    assert len(received) == 1
    assert received[0][1] is True and received[0][2] == b"FAKE_JIT_ARCHIVE"
    assert fake_torch.jit.load is fake_orig_load

"""Verify proxy configuration without downloads, installations or services."""

from typing import Any

import pytest
from yt_dlp import YoutubeDL

from core import _1_ytdlp

PROXY_DISCOVERY = YoutubeDL.proxies.func.__globals__["urllib"].request


def options(tmp_path, monkeypatch, youtube) -> Any:
    recorded: dict[str, Any] = {}

    class Download:
        def __init__(self, opts):
            recorded.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def download(self, urls):
            assert urls == ["https://video.example.com/item"]

    monkeypatch.setattr(
        _1_ytdlp, "load_key", lambda key: youtube if key == "youtube" else ""
    )
    monkeypatch.setattr(_1_ytdlp, "update_ytdlp", lambda: Download)
    monkeypatch.setattr(_1_ytdlp, "find_video_files", lambda _: "synthetic.mp4")
    monkeypatch.setattr(_1_ytdlp, "write_input_manifest", lambda *a: None)
    monkeypatch.setattr(_1_ytdlp, "resolve_task_path", lambda p: p)
    monkeypatch.setattr(_1_ytdlp, "TaskWorkspace", lambda *a: None)
    monkeypatch.setattr(_1_ytdlp, "probe_media_metadata", lambda *a: {})
    _1_ytdlp.download_video_ytdlp("https://video.example.com/item", str(tmp_path))
    return recorded


@pytest.mark.parametrize("config", [{}, {"proxy": None}])
def test_default_preserves_environment_discovery(tmp_path, monkeypatch, config):
    opts = options(tmp_path, monkeypatch, config)
    assert "proxy" not in opts
    monkeypatch.setattr(
        PROXY_DISCOVERY,
        "getproxies",
        lambda: {"https": "http://proxy.example.com:8080"},
    )
    with YoutubeDL({**opts, "quiet": True, "js_runtimes": {}}, auto_init=False) as ydl:  # type: ignore[arg-type]
        assert ydl.proxies["https"] == "http://proxy.example.com:8080"


def test_default_without_proxy_connects_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(PROXY_DISCOVERY, "getproxies", lambda: {})
    with YoutubeDL(options(tmp_path, monkeypatch, {}), auto_init=False) as ydl:
        assert ydl.proxies == {}


@pytest.mark.parametrize(
    "proxy", ["", "http://proxy.example.com:8080", "socks5://proxy.example.com:1080"]
)
def test_explicit_overrides_environment(tmp_path, monkeypatch, proxy):
    opts = options(tmp_path, monkeypatch, {"proxy": proxy})
    monkeypatch.setattr(
        PROXY_DISCOVERY,
        "getproxies",
        lambda: pytest.fail("Explicit mode must not discover proxies"),
    )
    with YoutubeDL(opts, auto_init=False) as ydl:
        assert ydl.proxies == {"all": proxy or "__noproxy__"}


@pytest.mark.parametrize("proxy", [False, 7897, []])
def test_invalid_type_rejected_before_download(tmp_path, monkeypatch, proxy):
    with pytest.raises(ValueError, match="youtube.proxy"):
        options(tmp_path, monkeypatch, {"proxy": proxy})

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

from core import tray


def test_create_tray_icon_image_renders_valid_rgba():
    img = tray.create_tray_icon_image(64)
    assert img.size == (64, 64)
    assert img.mode == "RGBA"


def test_server_controller_lifecycle_and_log_capture(tmp_path: Path):
    log_file = tmp_path / "server.log"
    controller = tray.ServerController(
        python_executable=sys.executable,
        log_path=log_file,
    )
    # Replace the command target temporarily by monkeypatching Popen
    calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            calls.append(cmd)
            self.terminated = False
            kwargs["stdout"].write(b"server started\n")
            kwargs["stdout"].flush()

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    orig_popen = tray.subprocess.Popen
    tray.subprocess.Popen = FakeProc  # type: ignore[assignment]
    try:
        assert controller.start() is True
        assert len(calls) == 1
        # Starting again while running is a no-op
        assert controller.start() is True
        assert len(calls) == 1
        assert controller.restart() is True
        assert len(calls) == 2
        controller.stop()
    finally:
        tray.subprocess.Popen = orig_popen  # type: ignore[assignment]

    assert log_file.read_bytes() == b"server started\nserver started\n"


def test_tray_main_single_instance_guard_opens_browser(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(tray, "is_port_open", lambda port, **kw: True)
    monkeypatch.setattr(tray, "open_webui", lambda url=None: opened.append(url or ""))
    assert tray.main() == 0
    assert opened == [tray.get_url()]


def test_tray_main_starts_icon_and_cleans_up_server(monkeypatch, tmp_path: Path):
    events: list[str] = []
    monkeypatch.setattr(tray, "is_port_open", lambda port, **kw: False)
    monkeypatch.setattr(
        tray, "wait_and_open_browser", lambda *a, **kw: events.append("wait_browser")
    )

    class FakeController:
        log_path = tmp_path / "server.log"

        def start(self):
            events.append("server_start")
            return True

        def stop(self):
            events.append("server_stop")

        def restart(self):
            events.append("server_restart")
            return True

    class FakeIcon:
        def __init__(self, name, icon, title, menu):
            self.name = name
            self.menu = menu

        def run(self):
            events.append("icon_run")

        def stop(self):
            events.append("icon_stop")

    class FakeMenu:
        SEPARATOR = "---"

        def __init__(self, *items: Any):
            self.items = items

    fake_pystray = SimpleNamespace(
        Menu=FakeMenu,
        MenuItem=lambda label, action, default=False: (label, action, default),
        Icon=FakeIcon,
    )

    monkeypatch.setattr(tray, "ServerController", FakeController)
    monkeypatch.setitem(sys.modules, "pystray", fake_pystray)

    assert tray.main() == 0
    assert events == ["server_start", "wait_browser", "icon_run", "server_stop"]

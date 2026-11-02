"""System tray launcher for VideoLingo WebUI (headless background server + tray controls)."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "output" / "server.log"


def get_port() -> int:
    return int(os.environ.get("VIDEOLINGO_API_PORT", "8501"))


def get_url() -> str:
    return f"http://127.0.0.1:{get_port()}"


def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def create_tray_icon_image(size: int = 64) -> Any:
    """Render a clean VideoLingo play-badge icon in memory using Pillow."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = 4
    draw.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=14,
        fill=(37, 99, 235, 255),
    )
    # White play triangle in the upper-center
    draw.polygon(
        [(24, 16), (24, 38), (44, 27)],
        fill=(255, 255, 255, 255),
    )
    # Gold subtitle bar at the bottom
    draw.rounded_rectangle(
        (16, 44, size - 16, 50),
        radius=3,
        fill=(250, 204, 21, 255),
    )
    return img


class ServerController:
    """Manage the child `api.py` server process and log file."""

    def __init__(self, python_executable: str | None = None, log_path: Path = LOG_PATH):
        self.python_executable = python_executable or sys.executable
        self.log_path = log_path
        self.proc: subprocess.Popen[Any] | None = None
        self._log_handle: Any = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return True
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._close_log()
            self._log_handle = self.log_path.open("ab")
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            kwargs: dict[str, Any] = {
                "cwd": str(ROOT),
                "stdout": self._log_handle,
                "stderr": subprocess.STDOUT,
                "stdin": subprocess.DEVNULL,
                "env": env,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NO_WINDOW", 0x08000000
                )
            self.proc = subprocess.Popen(
                [self.python_executable, str(ROOT / "api.py")],
                **kwargs,
            )
            return True

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            proc = self.proc
            self.proc = None
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            self._close_log()

    def restart(self) -> bool:
        self.stop()
        return self.start()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None


def open_webui(url: str | None = None) -> None:
    webbrowser.open(url or get_url())


def open_log_file(path: Path = LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def wait_and_open_browser(port: int, url: str, timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_port_open(port):
            open_webui(url)
            return
        time.sleep(0.2)


def main() -> int:
    port = get_port()
    url = get_url()

    # Single-instance guard: if the server is already listening on the port, just open the browser and exit.
    if is_port_open(port):
        open_webui(url)
        return 0

    import pystray  # type: ignore[import-untyped]

    controller = ServerController()
    controller.start()
    threading.Thread(
        target=wait_and_open_browser,
        args=(port, url),
        daemon=True,
    ).start()

    def on_open(icon: Any, item: Any = None) -> None:
        open_webui(url)

    def on_logs(icon: Any, item: Any = None) -> None:
        open_log_file(controller.log_path)

    def on_restart(icon: Any, item: Any = None) -> None:
        controller.restart()
        threading.Thread(
            target=wait_and_open_browser,
            args=(port, url),
            daemon=True,
        ).start()

    def on_quit(icon: Any, item: Any = None) -> None:
        controller.stop()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("打开 VideoLingo 网页", on_open, default=True),
        pystray.MenuItem("查看运行日志", on_logs),
        pystray.MenuItem("重启后端服务", on_restart),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", on_quit),
    )
    icon = pystray.Icon(
        "VideoLingo",
        create_tray_icon_image(),
        f"VideoLingo ({url})",
        menu,
    )
    try:
        icon.run()
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

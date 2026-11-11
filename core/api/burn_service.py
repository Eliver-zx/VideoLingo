from __future__ import annotations

from collections import deque
import contextlib
import datetime
from pathlib import Path
import subprocess
import threading
from threading import Event, Lock, Thread
from typing import Any, Callable

from fastapi import HTTPException

from core.task_runner import StopTask


class StandaloneBurnJob:
    def __init__(
        self, broadcast_fn: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        self._lock = Lock()
        self._stop_event = Event()
        self._process: subprocess.Popen[str] | None = None
        self._thread: Thread | None = None
        self._broadcast_fn = broadcast_fn
        self.state: str = "idle"
        self.progress: float = 0.0
        self.sub_step: str = ""
        self.video_path: str = ""
        self.subtitle_path: str = ""
        self.secondary_subtitle_path: str = ""
        self.output_path: str = ""
        self.completed_output_path: str = ""
        self.error: str = ""
        self._logs: deque[dict[str, Any]] = deque(maxlen=300)

    def _record_log(
        self, msg: str, tag: str = "asr", label: str = "VIDEO", warn: bool = False
    ) -> dict[str, Any]:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        entry = {
            "time": now_str,
            "tag": "warn" if warn else tag,
            "label": label.upper()[:8],
            "msg": msg,
            "warn": bool(warn),
        }
        self._logs.append(entry)
        return entry

    def _status_fields_unlocked(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "progress": round(self.progress, 4),
            "sub_step": self.sub_step,
            "video_path": self.video_path,
            "subtitle_path": self.subtitle_path,
            "secondary_subtitle_path": self.secondary_subtitle_path,
            "output_path": self.output_path,
            "completed_output_path": self.completed_output_path,
            "error": self.error,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._status_fields_unlocked(),
                "logs": list(self._logs),
            }

    def _emit(self, log_entry: dict[str, Any] | None = None) -> None:
        with self._lock:
            payload: dict[str, Any] = {
                "type": "standalone_burn",
                **self._status_fields_unlocked(),
            }
        if log_entry is not None:
            payload["log"] = log_entry
        if self._broadcast_fn is not None:
            self._broadcast_fn(payload)
        else:
            import api as _api

            _api._broadcast_sse(payload)

    def _register_process(self, proc: subprocess.Popen[str] | None) -> None:
        with self._lock:
            self._process = proc

    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(
                self.state == "running" and self._thread and self._thread.is_alive()
            )

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        with self._lock:
            proc = self._process
            worker = self._thread
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=0.5)
        return self.snapshot()

    def start(
        self,
        video_path: Path,
        subtitle_path: Path,
        secondary_subtitle_path: Path | None,
        output_path: Path,
        ffmpeg_gpu: bool | None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.state == "running" and self._thread and self._thread.is_alive():
                raise HTTPException(
                    409, "当前已有烧录任务正在运行，请等待完成或先停止。"
                )
            self._stop_event.clear()
            self._logs.clear()
            self.state = "running"
            self.progress = 0.02
            self.sub_step = "准备压制…"
            self.video_path = str(video_path)
            self.subtitle_path = str(subtitle_path)
            self.secondary_subtitle_path = (
                str(secondary_subtitle_path) if secondary_subtitle_path else ""
            )
            self.output_path = str(output_path)
            self.completed_output_path = ""
            self.error = ""
            init_log = self._record_log(
                f"准备烧录视频: {video_path.name} -> {output_path}",
                tag="asr",
                label="BURN",
            )

        self._emit(init_log)

        def _worker() -> None:
            from core._7_sub_into_vid import burn_subtitles_to_video
            from core.utils.subtitle_style import build_standalone_subtitle_filter

            def _on_progress(frac: float, sub_lbl: str, msg: str | None) -> None:
                log_entry = None
                with self._lock:
                    self.progress = frac
                    self.sub_step = sub_lbl
                    if msg:
                        log_entry = self._record_log(msg, tag="asr", label="VIDEO")
                self._emit(log_entry)

            def _on_log(msg: str, tag: str, label: str, warn: bool) -> None:
                with self._lock:
                    log_entry = self._record_log(msg, tag=tag, label=label, warn=warn)
                self._emit(log_entry)

            preexisting_sig: tuple[int, int] | None = None
            with contextlib.suppress(OSError):
                if output_path.is_file():
                    st = output_path.stat()
                    preexisting_sig = (st.st_mtime_ns, st.st_size)

            def _cleanup_partial_output() -> None:
                with contextlib.suppress(OSError):
                    if not output_path.is_file():
                        return
                    if preexisting_sig is not None:
                        cur_st = output_path.stat()
                        if (cur_st.st_mtime_ns, cur_st.st_size) == preexisting_sig:
                            return
                    output_path.unlink(missing_ok=True)

            try:
                vf_filter = build_standalone_subtitle_filter(
                    subtitle_path, secondary_subtitle_path
                )
                elapsed = burn_subtitles_to_video(
                    str(video_path),
                    str(output_path),
                    vf_filter,
                    ffmpeg_gpu=ffmpeg_gpu,
                    on_progress=_on_progress,
                    on_log=_on_log,
                    should_stop=self._stop_event.is_set,
                    register_process=self._register_process,
                )
                with self._lock:
                    self.state = "completed"
                    self.progress = 1.0
                    self.sub_step = f"烧录完成 ({elapsed:.1f}s)"
                    self.completed_output_path = str(output_path)
                    done_log = self._record_log(
                        f"烧录完成，耗时 {elapsed:.1f}s -> {output_path}",
                        tag="asr",
                        label="DONE",
                    )
                self._emit(done_log)
            except StopTask:
                _cleanup_partial_output()
                with self._lock:
                    self.state = "stopped"
                    self.sub_step = "已手动停止"
                    stop_log = self._record_log(
                        "已停止当前视频烧录。", tag="warn", label="STOP", warn=True
                    )
                self._emit(stop_log)
            except Exception as exc:
                _cleanup_partial_output()
                err_log = None
                with self._lock:
                    self.state = "error"
                    self.error = str(exc)
                    self.sub_step = "烧录失败"
                    if not self._logs or not self._logs[-1].get("warn"):
                        err_log = self._record_log(
                            f"烧录失败: {exc}", tag="warn", label="ERROR", warn=True
                        )
                self._emit(err_log)

        t = Thread(target=_worker, daemon=True)
        with self._lock:
            self._thread = t
        t.start()
        return self.snapshot()

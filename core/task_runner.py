"""Single-process sequential runner and serial queue shared by Web UI and HTTP API."""

from __future__ import annotations

from collections import deque
import contextlib
import contextvars
from dataclasses import dataclass, field
import datetime
import re
import threading
import time
import traceback
from typing import Any, Callable, ClassVar, Iterator, Literal


ArtifactKind = Literal["words", "terms", "utterances", "subtitles", "video"]
LogTag = Literal["asr", "nlp", "llm", "warn"]
LogLevel = Literal["info", "warn", "error"]

_MARKUP_RE = re.compile(r"\[/?[a-zA-Z0-9_#\s.,=-]+\]")
_LEADING_ICON_RE = re.compile(r"^[⚠️❌✅🎬🎙️🚀🔍📦📊💡⏳🎉🔥✨]+\s*")


def clean_log_text(text: str | None) -> str:
    """Strip any Rich-style bracket tags and collapse internal newlines into a single line."""
    if not text:
        return ""
    stripped = _LEADING_ICON_RE.sub("", _MARKUP_RE.sub("", str(text)).strip())
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) > 2 and any(line.startswith(("{", "[")) for line in lines[1:]):
        return lines[0][:240]
    joined = " ".join(lines)
    return joined[:240] + ("…" if len(joined) > 240 else "")


def infer_step_tag_and_label(step_label: str) -> tuple[LogTag, str]:
    lower = (step_label or "").lower()
    if "transcription" in lower:
        return ("asr", "ASR")
    if "segmentation" in lower:
        return ("nlp", "NLP")
    if "summarization" in lower or "translation" in lower:
        return ("llm", "TRANS")
    if "cutting" in lower or "aligning" in lower or "subtitle" in lower:
        return ("llm", "SUBS")
    if "audio" in lower or "dub" in lower or "tts" in lower:
        return ("asr", "TTS")
    return ("asr", "RUN")


@dataclass(frozen=True)
class ProgressScope:
    """Affine interval [lo, hi] mapping local [0.0, 1.0] progress and default log metadata."""

    lo: float = 0.0
    hi: float = 1.0
    prefix: str = ""
    tag: LogTag = "asr"
    label: str = "RUN"

    def child(
        self,
        child_lo: float,
        child_hi: float,
        child_prefix: str = "",
        *,
        tag: LogTag | None = None,
        label: str | None = None,
    ) -> "ProgressScope":
        c_lo = max(0.0, min(1.0, float(child_lo)))
        c_hi = max(c_lo, min(1.0, float(child_hi)))
        span = self.hi - self.lo
        prefix = (
            f"{self.prefix} · {child_prefix}"
            if (self.prefix and child_prefix)
            else (child_prefix or self.prefix)
        )
        return ProgressScope(
            lo=self.lo + span * c_lo,
            hi=self.lo + span * c_hi,
            prefix=prefix,
            tag=tag or self.tag,
            label=label or self.label,
        )

    def map_fraction(self, local_fraction: float) -> float:
        clamped = max(0.0, min(1.0, float(local_fraction)))
        return self.lo + (self.hi - self.lo) * clamped

    def map_label(self, sub_label: str | None) -> str | None:
        if not sub_label:
            return self.prefix or None
        text = str(sub_label)
        if self.prefix and not text.startswith(self.prefix):
            return f"{self.prefix} · {text}"
        return text


_PROGRESS_SCOPE: contextvars.ContextVar[ProgressScope] = contextvars.ContextVar(
    "videolingo_progress_scope",
    default=ProgressScope(),
)


@contextlib.contextmanager
def progress_scope(
    lo: float = 0.0,
    hi: float = 1.0,
    prefix: str = "",
    *,
    tag: LogTag | None = None,
    label: str | None = None,
) -> Iterator[ProgressScope]:
    """Map nested report_progress([0.0..1.0]) calls into [lo, hi] of the parent scope."""
    parent = _PROGRESS_SCOPE.get()
    current = parent.child(lo, hi, prefix, tag=tag, label=label)
    token = _PROGRESS_SCOPE.set(current)
    try:
        yield current
    finally:
        _PROGRESS_SCOPE.reset(token)


class StopTask(Exception):
    """Raised when the task is stopped by user."""

    pass


@dataclass
class TaskRunner:
    """Manages a background thread that executes a sequence of steps with pause/stop control and SSE notifications."""

    # Public read-only state
    state: str = (
        "idle"  # idle | running | paused | stopping | stopped | completed | error
    )
    current_step: int = -1  # 0-indexed, -1 = not started
    total_steps: int = 0
    current_label: str = ""
    sub_label: str = ""
    sub_progress: float = 0.0
    error_msg: str = ""
    pause_message: str = ""  # why the task paused itself, empty when the user paused it
    active_task_id: str | None = None
    stages_done: dict[str, bool] = field(default_factory=dict)

    # Internal
    _pause_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _steps: list = field(default_factory=list)
    _listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)
    _queue: deque[
        tuple[str | None, list[tuple[str, Callable]], Callable[[], Any] | None]
    ] = field(default_factory=deque)
    _queue_lock: threading.Lock = field(default_factory=threading.Lock)
    _state_lock: threading.RLock = field(default_factory=threading.RLock)
    _logs: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=300))
    _last_log_msg: str = ""
    _last_log_ts: float = 0.0

    _current: ClassVar["TaskRunner | None"] = None

    def __post_init__(self):
        self._pause_event.set()  # not paused initially

    def add_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def recent_logs(self) -> list[dict[str, Any]]:
        with self._state_lock:
            return list(self._logs)

    def _notify(
        self,
        event_type: str = "status",
        message: str | None = None,
        *,
        artifact: ArtifactKind | None = None,
        log_tag: LogTag | None = None,
        log_label: str | None = None,
        log_level: LogLevel = "info",
    ) -> None:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        effective_msg = clean_log_text(message) if message else None
        default_tag, default_label = infer_step_tag_and_label(self.current_label)
        is_warn = bool(self.error_msg and event_type == "task_error") or log_level in (
            "warn",
            "error",
        )
        resolved_tag: LogTag = "warn" if is_warn else (log_tag or default_tag)
        resolved_label = (log_label or default_label or "RUN").upper()[:8]

        with self._state_lock:
            display_msg = (
                effective_msg
                or (self.pause_message if event_type == "task_paused" else "")
                or (self.error_msg if event_type == "task_error" else "")
            )
            if display_msg:
                self._logs.append(
                    {
                        "time": now_str,
                        "tag": resolved_tag,
                        "label": resolved_label,
                        "msg": display_msg,
                        "warn": is_warn,
                        "task_id": self.active_task_id,
                    }
                )
            stages_snapshot = dict(self.stages_done) if self.stages_done else None

        payload = {
            "type": event_type,
            "task_id": self.active_task_id,
            "state": self.state,
            "active": self.is_active,
            "step": self.current_label or None,
            "sub_step": self.sub_label or None,
            "sub_progress": round(self.sub_progress, 4),
            "step_index": self.current_step,
            "total_steps": self.total_steps,
            "progress": round(self.progress, 4),
            "message": effective_msg or None,
            "error": self.error_msg or None,
            "pause_message": self.pause_message or None,
            "queued_count": len(self._queue),
            "artifact": artifact,
            "stages_done": stages_snapshot,
            "log_tag": resolved_tag,
            "log_label": resolved_label,
            "log_level": log_level,
            "warn": is_warn,
            "time": now_str,
        }
        for cb in list(self._listeners):
            try:
                cb(payload)
            except Exception:
                pass

    # ------ Cancellation, progress & structured logging helpers ------

    @classmethod
    def check_cancel(cls) -> None:
        """Block while paused and raise :class:`StopTask` if a stop was requested."""
        runner = cls._current
        if runner is None:
            return
        runner._pause_event.wait()
        if runner._stop_event.is_set():
            raise StopTask()

    @classmethod
    def report_progress(
        cls,
        fraction: float | None = None,
        sub_label: str | None = None,
        message: str | None = None,
        *,
        tag: LogTag | None = None,
        label: str | None = None,
        level: LogLevel = "info",
    ) -> None:
        """Update sub-step progress (0.0 to 1.0) through the active ProgressScope and broadcast an SSE event."""
        runner = cls._current
        if runner is None:
            return
        scope = _PROGRESS_SCOPE.get()
        cleaned_msg = clean_log_text(message) if message else None
        resolved_tag: LogTag = (
            "warn" if level in ("warn", "error") else (tag or scope.tag)
        )
        resolved_label = (label or scope.label or "RUN").upper()[:8]
        emit_msg = cleaned_msg
        with runner._state_lock:
            prev_progress = round(runner.sub_progress, 4)
            prev_label = runner.sub_label
            if fraction is not None:
                runner.sub_progress = scope.map_fraction(fraction)
            if sub_label is not None:
                mapped = scope.map_label(sub_label)
                if mapped is not None:
                    runner.sub_label = mapped
            if cleaned_msg:
                now = time.monotonic()
                if (
                    cleaned_msg == runner._last_log_msg
                    and (now - runner._last_log_ts) < 0.25
                    and level == "info"
                ):
                    emit_msg = None
                else:
                    runner._last_log_msg = cleaned_msg
                    runner._last_log_ts = now
            if (
                emit_msg is None
                and round(runner.sub_progress, 4) == prev_progress
                and runner.sub_label == prev_label
                and prev_label != ""
            ):
                return
        runner._notify(
            "task_progress",
            message=emit_msg,
            log_tag=resolved_tag,
            log_label=resolved_label,
            log_level=level,
        )

    @classmethod
    def log_event(
        cls,
        message: str,
        *,
        tag: LogTag | None = None,
        label: str | None = None,
        level: LogLevel = "info",
    ) -> None:
        """Emit a structured log line over SSE without altering sub_progress."""
        runner = cls._current
        if runner is None:
            return
        cleaned = clean_log_text(message)
        if not cleaned:
            return
        scope = _PROGRESS_SCOPE.get()
        resolved_tag: LogTag = (
            "warn" if level in ("warn", "error") else (tag or scope.tag)
        )
        resolved_label = (label or scope.label or "RUN").upper()[:8]
        with runner._state_lock:
            now = time.monotonic()
            if (
                cleaned == runner._last_log_msg
                and (now - runner._last_log_ts) < 0.25
                and level == "info"
            ):
                return
            runner._last_log_msg = cleaned
            runner._last_log_ts = now
        runner._notify(
            "task_log",
            message=cleaned,
            log_tag=resolved_tag,
            log_label=resolved_label,
            log_level=level,
        )

    @classmethod
    def notify_artifact(
        cls,
        artifact: ArtifactKind,
        stages_done: dict[str, bool],
        message: str | None = None,
        *,
        tag: LogTag | None = None,
        label: str | None = None,
    ) -> None:
        """Broadcast that a workspace artifact table or output file was committed."""
        runner = cls._current
        if runner is None:
            return
        scope = _PROGRESS_SCOPE.get()
        cleaned = clean_log_text(message) if message else None
        resolved_tag: LogTag = tag or scope.tag
        resolved_label = (label or scope.label or "DATA").upper()[:8]
        with runner._state_lock:
            runner.stages_done = dict(stages_done)
            if cleaned:
                runner._last_log_msg = cleaned
                runner._last_log_ts = time.monotonic()
        runner._notify(
            "task_artifact",
            message=cleaned,
            artifact=artifact,
            log_tag=resolved_tag,
            log_label=resolved_label,
            log_level="info",
        )

    # ------ Control API ------

    def start(
        self,
        steps: list[tuple[str, Callable]],
        *,
        task_id: str | None = None,
        on_finish: Callable[[], Any] | None = None,
    ):
        """Start executing steps in a background thread."""
        if self.is_active:
            raise RuntimeError("A task is already active")

        with self._state_lock:
            self.active_task_id = task_id
            self._steps = steps
            self.total_steps = len(steps)
            self.current_step = -1
            self.current_label = ""
            self.sub_label = ""
            self.sub_progress = 0.0
            self.error_msg = ""
            self.pause_message = ""
            self.stages_done = {}
            self._last_log_msg = ""
            self._last_log_ts = 0.0
            self.state = "running"

        self._pause_event.set()
        self._stop_event.clear()

        self._notify("task_started", log_tag="asr", log_label="RUN")
        self._thread = threading.Thread(
            target=self._run, args=(on_finish,), daemon=True
        )
        self._thread.start()

    def enqueue(
        self,
        steps: list[tuple[str, Callable]],
        *,
        task_id: str | None = None,
        on_finish: Callable[[], Any] | None = None,
    ) -> bool:
        """Start immediately if idle, or append to the serial queue if a task is currently running."""
        with self._queue_lock:
            if not self.is_active:
                self.start(steps, task_id=task_id, on_finish=on_finish)
                return True
            self._queue.append((task_id, steps, on_finish))
            self._notify("queue_updated")
            return False

    def pause(self, message: str = ""):
        if self.state == "running":
            self._pause_event.clear()
            self.pause_message = message
            self.state = "paused"
            self._notify("task_paused")

    def resume(self):
        if self.state == "paused":
            self.pause_message = ""
            self._pause_event.set()
            self.state = "running"
            self._notify("task_resumed")

    def stop(self, *, clear_queue: bool = False):
        """Request stop for the active task; optionally clear queued tasks."""
        if clear_queue:
            with self._queue_lock:
                self._queue.clear()
        if self.state in ("running", "paused"):
            self.state = "stopping"
            self.pause_message = ""
            self._stop_event.set()
            self._pause_event.set()  # unblock if paused so thread can exit
            self._notify("task_stopping")

    def is_queued(self, task_id: str) -> bool:
        """Return True if task_id is currently waiting in the serial queue."""
        with self._queue_lock:
            return any(item[0] == task_id for item in self._queue)

    def cancel_queued(self, task_id: str) -> bool:
        """Remove a queued task by task_id before it starts."""
        with self._queue_lock:
            for item in list(self._queue):
                if item[0] == task_id:
                    self._queue.remove(item)
                    self._notify("queue_updated")
                    return True
        return False

    def reset(self):
        """Reset to idle state (only when not running)."""
        if not self.is_active:
            with self._state_lock:
                self.state = "idle"
                self.active_task_id = None
                self.current_step = -1
                self.total_steps = 0
                self.current_label = ""
                self.sub_label = ""
                self.sub_progress = 0.0
                self.error_msg = ""
                self.pause_message = ""
                self.stages_done = {}
                self._last_log_msg = ""
                self._last_log_ts = 0.0
                self._steps = []
            self._notify("task_reset")

    @property
    def is_active(self) -> bool:
        return self.state in ("running", "paused", "stopping") or bool(
            self._thread and self._thread.is_alive()
        )

    @property
    def is_done(self) -> bool:
        return self.state in ("completed", "stopped", "error")

    @property
    def progress(self) -> float:
        """0.0 to 1.0"""
        if self.total_steps == 0:
            return 0.0
        if self.state == "completed":
            return 1.0
        base = max(self.current_step, 0)
        sub = max(0.0, min(1.0, self.sub_progress)) if self.current_step >= 0 else 0.0
        return min(0.99, (base + sub) / self.total_steps)

    # ------ Internal ------

    def _run(self, on_finish: Callable[[], Any] | None = None):
        """Execute steps sequentially in background thread, then advance serial queue if any."""
        type(self)._current = self
        try:
            for i, (label, func) in enumerate(self._steps):
                if self._stop_event.is_set():
                    self.state = "stopped"
                    self._notify("task_stopped")
                    return

                self._pause_event.wait()

                if self._stop_event.is_set():
                    self.state = "stopped"
                    self._notify("task_stopped")
                    return

                step_tag, step_label = infer_step_tag_and_label(label)
                with self._state_lock:
                    self.current_step = i
                    self.current_label = label
                    self.sub_label = ""
                    self.sub_progress = 0.0
                token = _PROGRESS_SCOPE.set(
                    ProgressScope(
                        lo=0.0, hi=1.0, prefix="", tag=step_tag, label=step_label
                    )
                )
                try:
                    self._notify(
                        "task_progress",
                        message=f"开始阶段 [{i + 1}/{self.total_steps}]: {label}",
                        log_tag=step_tag,
                        log_label=step_label,
                    )
                    func()
                    self.check_cancel()
                finally:
                    _PROGRESS_SCOPE.reset(token)

            self.state = "completed"
            self._notify(
                "task_completed",
                message="全部流水线阶段执行完成",
                log_tag="llm",
                log_label="DONE",
            )
        except StopTask:
            self.state = "stopped"
            self._notify(
                "task_stopped",
                message="任务已停止",
                log_tag="warn",
                log_label="STOP",
                log_level="warn",
            )
        except Exception as e:
            self.error_msg = str(e)
            self.state = "error"
            self._notify(
                "task_error",
                log_tag="warn",
                log_label="ERROR",
                log_level="error",
            )
            traceback.print_exc()
        finally:
            if on_finish is not None:
                try:
                    on_finish()
                except Exception:
                    pass
            if type(self)._current is self:
                type(self)._current = None
            self._drain_next_queued()

    def _drain_next_queued(self) -> None:
        with self._queue_lock:
            if not self._queue:
                return
            next_task_id, next_steps, next_finish = self._queue.popleft()
            self._thread = None
            self.state = "idle"
        self.start(next_steps, task_id=next_task_id, on_finish=next_finish)

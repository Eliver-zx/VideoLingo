"""Single-process sequential runner and serial queue shared by Web UI and HTTP API."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import traceback
from typing import Any, Callable, ClassVar


class StopTask(Exception):
    """Raised when the task is stopped by user."""

    pass


@dataclass
class TaskRunner:
    """Manages a background thread that executes a sequence of steps with pause/stop control and SSE notifications."""

    # Public read-only state
    state: str = "idle"  # idle | running | paused | stopping | stopped | completed | error
    current_step: int = -1  # 0-indexed, -1 = not started
    total_steps: int = 0
    current_label: str = ""
    error_msg: str = ""
    pause_message: str = ""  # why the task paused itself, empty when the user paused it
    active_task_id: str | None = None

    # Internal
    _pause_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _steps: list = field(default_factory=list)
    _listeners: list[Callable[[dict[str, Any]], None]] = field(default_factory=list)
    _queue: deque[tuple[str | None, list[tuple[str, Callable]], Callable[[], Any] | None]] = field(default_factory=deque)
    _queue_lock: threading.Lock = field(default_factory=threading.Lock)

    _current: ClassVar["TaskRunner | None"] = None

    def __post_init__(self):
        self._pause_event.set()  # not paused initially

    def add_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self, event_type: str = "status") -> None:
        payload = {
            "type": event_type,
            "task_id": self.active_task_id,
            "state": self.state,
            "active": self.is_active,
            "step": self.current_label or None,
            "step_index": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.progress,
            "error": self.error_msg or None,
            "pause_message": self.pause_message or None,
            "queued_count": len(self._queue),
        }
        for cb in list(self._listeners):
            try:
                cb(payload)
            except Exception:
                pass

    # ------ Cancellation helpers (called from core code) ------

    @classmethod
    def check_cancel(cls) -> None:
        """Block while paused and raise :class:`StopTask` if a stop was requested."""
        runner = cls._current
        if runner is None:
            return
        runner._pause_event.wait()
        if runner._stop_event.is_set():
            raise StopTask()

    # ------ Control API ------

    def start(self, steps: list[tuple[str, Callable]], *, task_id: str | None = None, on_finish: Callable[[], Any] | None = None):
        """Start executing steps in a background thread."""
        if self.is_active:
            raise RuntimeError("A task is already active")

        self.active_task_id = task_id
        self._steps = steps
        self.total_steps = len(steps)
        self.current_step = -1
        self.current_label = ""
        self.error_msg = ""
        self.pause_message = ""
        self.state = "running"

        self._pause_event.set()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run, args=(on_finish,), daemon=True)
        self._thread.start()
        self._notify("task_started")

    def enqueue(self, steps: list[tuple[str, Callable]], *, task_id: str | None = None, on_finish: Callable[[], Any] | None = None) -> bool:
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

    def stop(self):
        """Request stop. The task will halt before the next step."""
        with self._queue_lock:
            self._queue.clear()
        if self.state in ("running", "paused"):
            self.state = "stopping"
            self.pause_message = ""
            self._stop_event.set()
            self._pause_event.set()  # unblock if paused so thread can exit
            self._notify("task_stopping")

    def reset(self):
        """Reset to idle state (only when not running)."""
        if not self.is_active:
            self.state = "idle"
            self.active_task_id = None
            self.current_step = -1
            self.total_steps = 0
            self.current_label = ""
            self.error_msg = ""
            self.pause_message = ""
            self._steps = []
            self._notify("task_reset")

    @property
    def is_active(self) -> bool:
        return self.state in ("running", "paused", "stopping") or bool(self._thread and self._thread.is_alive())

    @property
    def is_done(self) -> bool:
        return self.state in ("completed", "stopped", "error")

    @property
    def progress(self) -> float:
        """0.0 to 1.0"""
        if self.total_steps == 0:
            return 0.0
        return 1.0 if self.state == "completed" else max(self.current_step, 0) / self.total_steps

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

                self.current_step = i
                self.current_label = label
                self._notify("task_progress")
                func()
                self.check_cancel()

            self.state = "completed"
            self._notify("task_completed")
        except StopTask:
            self.state = "stopped"
            self._notify("task_stopped")
        except Exception as e:
            self.error_msg = str(e)
            self.state = "error"
            self._notify("task_error")
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
            if not self._queue or self.state == "stopped":
                return
            next_task_id, next_steps, next_finish = self._queue.popleft()
            self._thread = None
        self.start(next_steps, task_id=next_task_id, on_finish=next_finish)

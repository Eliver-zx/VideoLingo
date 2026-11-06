from __future__ import annotations

from typing import Any


def check_cancel() -> None:
    """Cooperative cancellation hook for long-running core loops."""
    from core.task_runner import TaskRunner

    TaskRunner.check_cancel()


def report_progress(
    fraction: float | None = None,
    sub_label: str | None = None,
    message: str | None = None,
    *,
    tag: Any = None,
    label: str | None = None,
    level: Any = "info",
) -> None:
    """Report sub-step progress (0.0 to 1.0) and optional structured log message to the active TaskRunner."""
    from core.task_runner import TaskRunner

    TaskRunner.report_progress(
        fraction=fraction,
        sub_label=sub_label,
        message=message,
        tag=tag,
        label=label,
        level=level,
    )


def log_event(
    message: str,
    *,
    tag: Any = None,
    label: str | None = None,
    level: Any = "info",
) -> None:
    """Emit a structured log line to the active TaskRunner SSE stream."""
    from core.task_runner import TaskRunner

    TaskRunner.log_event(message, tag=tag, label=label, level=level)


def progress_scope(
    lo: float = 0.0,
    hi: float = 1.0,
    prefix: str = "",
    *,
    tag: Any = None,
    label: str | None = None,
):
    """Map nested report_progress([0.0..1.0]) calls into [lo, hi] of the parent scope."""
    from core.task_runner import progress_scope as _scope

    return _scope(lo, hi, prefix, tag=tag, label=label)


# use try-except to avoid error when installing
_import_error = None
try:
    from .ask_gpt import ask_gpt
    from .decorator import except_handler, check_file_exists
    from .config_utils import (
        load_key,
        load_key_or,
        update_key,
        set_runtime_state,
        get_joiner,
        join_words,
        get_source_language,
    )
except ImportError as e:
    _import_error = e


def __getattr__(name):
    # Report the import that really failed instead of "module has no attribute ..."
    if _import_error is not None and name in __all__:
        raise ImportError(
            f"core.utils could not be loaded: {_import_error}. "
            "Some dependencies are missing; run Install.command / Install.bat or `uv pip install -r requirements.txt`."
        ) from _import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ask_gpt",
    "except_handler",
    "check_file_exists",
    "load_key",
    "load_key_or",
    "update_key",
    "set_runtime_state",
    "get_joiner",
    "join_words",
    "get_source_language",
    "check_cancel",
    "report_progress",
    "log_event",
    "progress_scope",
]

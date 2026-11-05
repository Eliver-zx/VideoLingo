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
    from rich import print as rprint
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


def check_cancel():
    """Cooperative cancellation hook for long-running core loops.

    Uses the shared runner when a pipeline is active; otherwise a no-op.
    """
    from core.task_runner import TaskRunner

    TaskRunner.check_cancel()


def report_progress(
    fraction: float | None = None,
    sub_label: str | None = None,
    message: str | None = None,
) -> None:
    """Report sub-step progress (0.0 to 1.0) and optional log message to the active TaskRunner."""
    from core.task_runner import TaskRunner

    TaskRunner.report_progress(fraction=fraction, sub_label=sub_label, message=message)


__all__ = [
    "ask_gpt",
    "except_handler",
    "check_file_exists",
    "load_key",
    "load_key_or",
    "update_key",
    "set_runtime_state",
    "rprint",
    "get_joiner",
    "join_words",
    "get_source_language",
    "check_cancel",
    "report_progress",
]

from core.task_runner import StopTask
from core.utils.models import resolve_task_path

import functools
import time
import os
from rich import print as rprint

_PATH_TO_GLOBAL_ATTR = {
    "output/log/cleaned_chunks.xlsx": "_2_CLEANED_CHUNKS",
    "output/log/split_by_nlp.txt": "_3_1_SPLIT_BY_NLP",
    "output/log/split_by_meaning.txt": "_3_2_SPLIT_BY_MEANING",
    "output/log/terminology.json": "_4_1_TERMINOLOGY",
    "output/log/translation_results.xlsx": "_4_2_TRANSLATION",
    "output/log/translation_results_for_subtitles.xlsx": "_5_SPLIT_SUB",
    "output/log/translation_results_remerged.xlsx": "_5_REMERGED",
    "output/audio/tts_tasks.xlsx": "_8_1_AUDIO_TASK",
}

# ------------------------------
# retry decorator
# ------------------------------


def except_handler(error_msg, retry=0, delay=1, default_return=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retry + 1):
                try:
                    return func(*args, **kwargs)
                except StopTask:
                    raise
                except Exception as e:
                    if isinstance(e, ValueError) and str(e) == "API key is not set":
                        raise
                    last_exception = e
                    rprint(f"[red]{error_msg}: {e}, retry: {i + 1}/{retry}[/red]")
                    if i == retry:
                        if default_return is not None:
                            return default_return
                        raise last_exception
                    time.sleep(delay * (2**i))

        return wrapper

    return decorator


# ------------------------------
# check file exists decorator
# ------------------------------


def check_file_exists(file_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from core.workspace import workspace_for_path

            attr_name = _PATH_TO_GLOBAL_ATTR.get(str(file_path))
            effective = (
                func.__globals__.get(attr_name, file_path) if attr_name else file_path
            )
            resolved = resolve_task_path(effective)
            ws = workspace_for_path(resolved)
            if os.path.exists(resolved) or ws.is_step_done(resolved):
                rprint(
                    f"[yellow]⚠️ Step <{resolved}> already completed, skip <{func.__name__}> step.[/yellow]"
                )
                return
            return func(*args, **kwargs)

        return wrapper

    return decorator

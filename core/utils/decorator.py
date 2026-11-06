from core.task_runner import StopTask, TaskRunner
from core.utils.models import resolve_task_path

import functools
import time
import os


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
                    TaskRunner.log_event(
                        f"{error_msg}: {e} (重试 {i + 1}/{retry})",
                        tag="warn",
                        label="RETRY",
                        level="warn",
                    )
                    if i == retry:
                        if default_return is not None:
                            return default_return
                        raise last_exception
                    time.sleep(delay * (2**i))

        return wrapper

    return decorator


# ------------------------------
# check step / file completion decorator
# ------------------------------


def check_file_exists(step_or_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from core.workspace import get_active_workspace

            ws = get_active_workspace()
            if ws.is_step_done(step_or_path):
                TaskRunner.log_event(
                    f"阶段 <{step_or_path}> 已完成，跳过 {func.__name__}",
                    label="CACHE",
                )
                return
            resolved = resolve_task_path(step_or_path)
            if os.path.exists(resolved):
                TaskRunner.log_event(
                    f"产物 <{os.path.basename(resolved)}> 已存在，跳过 {func.__name__}",
                    label="CACHE",
                )
                return
            return func(*args, **kwargs)

        return wrapper

    return decorator

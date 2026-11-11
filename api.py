"""Local Web UI + SSE + REST API server for VideoLingo."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from threading import Event, Lock
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from core.api.burn_service import StandaloneBurnJob
from core.api.helpers import (
    _clean_user_path,
    _resolve_workspace,
    _workspace_summary,
    _wrap_workspace_steps,
    subtitle_file,
)
from core.api.routes_burn import router as burn_router
from core.api.routes_glossary import router as glossary_router
from core.api.routes_system import (
    WEBUI_DIR,
    WEBUI_HTML_PATH,
    check_llm_api,
    router as system_router,
)
from core.api.routes_tasks import router as tasks_router
from core.api.routes_terms import router as terms_router
from core.api.routes_utterances import router as utterances_router
from core.api.schemas import (
    ConfigUpdateRequest,
    CreateDomainRequest,
    CreateTaskRequest,
    DetectMaiRegionRequest,
    DomainTermSyncRequest,
    InvalidateStageRequest,
    LLMProbeRequest,
    PickFilesRequest,
    RunRequest,
    StandaloneBurnRequest,
    TaskLogEntryRequest,
    TermsUpdateRequest,
    UtterancesUpdateRequest,
)
from core.pipeline import get_steps
from core.task_runner import TaskRunner
from core.workspace import TaskWorkspace


def _configure_utf8_console() -> None:
    """Allow Rich and task threads to print Unicode on Windows and sanitize PATH for FFmpeg discovery."""
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    from core.utils.models import ensure_windows_ffmpeg_path

    ensure_windows_ffmpeg_path()


_configure_utf8_console()

app = FastAPI(
    title="VideoLingo",
    description="Serial multi-task workspace with SSE and zero-copy local media import.",
)
runner = TaskRunner()
operation_lock = Lock()
OUTPUT = Path("output")

_sse_subscribers: list[
    tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]
] = []
_sse_lock = Lock()
_shutdown_event = Event()


def _push_sse_queue(
    q: asyncio.Queue[dict[str, Any]],
    payload: dict[str, Any],
    force_overwrite: bool = False,
) -> None:
    try:
        q.put_nowait(payload)
    except asyncio.QueueFull:
        if force_overwrite:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


def _deliver_to_subscriber(
    loop: asyncio.AbstractEventLoop,
    q: asyncio.Queue[dict[str, Any]],
    payload: dict[str, Any],
    *,
    force_overwrite: bool = False,
) -> None:
    if loop.is_closed():
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _push_sse_queue(q, payload, force_overwrite)
    else:
        try:
            loop.call_soon_threadsafe(_push_sse_queue, q, payload, force_overwrite)
        except RuntimeError:
            pass


def _broadcast_sse(payload: dict[str, Any]) -> None:
    with _sse_lock:
        subs = list(_sse_subscribers)
    for loop, q in subs:
        _deliver_to_subscriber(loop, q, payload)


burn_job = StandaloneBurnJob()


def _trigger_shutdown() -> None:
    """Signal all active SSE streams and background runner threads to stop immediately."""
    _shutdown_event.set()
    runner.stop()
    burn_job.stop()
    with _sse_lock:
        subs = list(_sse_subscribers)
    for loop, q in subs:
        _deliver_to_subscriber(loop, q, {"type": "shutdown"}, force_overwrite=True)


def _on_runner_event(payload: dict[str, Any]) -> None:
    task_id = payload.get("task_id")
    ws = (
        TaskWorkspace(OUTPUT / task_id)
        if (task_id and (OUTPUT / task_id).is_dir())
        else None
    )
    if ws is not None:
        if payload.get("stages_done") is None and payload.get("type") in (
            "task_started",
            "task_paused",
            "task_resumed",
            "task_completed",
            "task_error",
            "task_stopped",
        ):
            try:
                stages_done = ws.compute_stages_done()
                payload["stages_done"] = stages_done
                if runner.active_task_id == task_id:
                    runner.stages_done = dict(stages_done)
            except OSError:
                pass
        ev_type = payload.get("type")
        msg = (
            payload.get("message")
            or (payload.get("pause_message") if ev_type == "task_paused" else None)
            or (payload.get("error") if ev_type == "task_error" else None)
        )
        if msg:
            ws.append_log(
                msg,
                tag=payload.get("log_tag") or "llm",
                label=payload.get("log_label") or "RUN",
                warn=bool(payload.get("warn")),
                time_str=payload.get("time"),
            )
        if ws.manifest_path.is_file() and payload.get("type") not in (
            "task_progress",
            "task_log",
            "task_artifact",
        ):
            try:
                ws.update_manifest(
                    execution={
                        "state": payload.get("state"),
                        "current_step": payload.get("step"),
                        "sub_step": payload.get("sub_step"),
                        "sub_progress": payload.get("sub_progress", 0.0),
                        "step_index": payload.get("step_index", -1),
                        "total_steps": payload.get("total_steps", 0),
                        "progress": payload.get("progress", 0.0),
                        "pause_reason": payload.get("pause_message"),
                        "error": payload.get("error"),
                    }
                )
            except OSError:
                pass
    _broadcast_sse(payload)


runner.add_listener(_on_runner_event)


@app.get("/api/events")
async def sse_events(request: Request):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
    sub_entry = (loop, q)
    with _sse_lock:
        _sse_subscribers.append(sub_entry)

    async def event_generator():
        try:
            init_payload = {
                "type": "connected",
                "state": runner.state,
                "active_task_id": runner.active_task_id,
                "progress": runner.progress,
                "stages_done": dict(runner.stages_done) if runner.stages_done else None,
                "recent_logs": (
                    runner.recent_logs(runner.active_task_id)
                    if runner.active_task_id
                    else []
                ),
            }
            yield f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
            while not _shutdown_event.is_set():
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    if _shutdown_event.is_set() or await request.is_disconnected():
                        break
                    yield 'data: {"type": "heartbeat"}\n\n'
                    continue
                if event.get("type") == "shutdown" or _shutdown_event.is_set():
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            with _sse_lock:
                if sub_entry in _sse_subscribers:
                    _sse_subscribers.remove(sub_entry)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.include_router(burn_router)
app.include_router(tasks_router)
app.include_router(terms_router)
app.include_router(utterances_router)
app.include_router(glossary_router)
app.include_router(system_router)


def _create_uvicorn_server(host: str = "127.0.0.1", port: int = 8501):
    import uvicorn

    _shutdown_event.clear()

    class _VideoLingoServer(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            _trigger_shutdown()
            super().handle_exit(sig, frame)

        async def shutdown(self, sockets: list[Any] | None = None) -> None:
            _trigger_shutdown()
            await super().shutdown(sockets=sockets)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=1,
    )
    return _VideoLingoServer(config)


__all__ = [
    "ConfigUpdateRequest",
    "CreateDomainRequest",
    "CreateTaskRequest",
    "DetectMaiRegionRequest",
    "DomainTermSyncRequest",
    "InvalidateStageRequest",
    "LLMProbeRequest",
    "OUTPUT",
    "PickFilesRequest",
    "RunRequest",
    "StandaloneBurnJob",
    "StandaloneBurnRequest",
    "TaskLogEntryRequest",
    "TermsUpdateRequest",
    "UtterancesUpdateRequest",
    "WEBUI_DIR",
    "WEBUI_HTML_PATH",
    "_broadcast_sse",
    "_clean_user_path",
    "_create_uvicorn_server",
    "_on_runner_event",
    "_resolve_workspace",
    "_shutdown_event",
    "_sse_subscribers",
    "_trigger_shutdown",
    "_workspace_summary",
    "_wrap_workspace_steps",
    "app",
    "burn_job",
    "check_llm_api",
    "get_steps",
    "operation_lock",
    "runner",
    "sse_events",
    "subtitle_file",
    "time",
]


if __name__ == "__main__":
    port = int(os.environ.get("VIDEOLINGO_API_PORT", "8501"))
    _create_uvicorn_server(host="127.0.0.1", port=port).run()

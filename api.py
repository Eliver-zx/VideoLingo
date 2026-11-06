"""Local Web UI + SSE + REST API server for VideoLingo."""

from __future__ import annotations

import asyncio
from functools import partial
import io
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
from threading import Event, Lock
import time
from typing import Any, Literal
from urllib.parse import quote, urlparse
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.pipeline import get_steps
from core.task_runner import TaskRunner
from core.utils.config_utils import (
    CONFIG_PATH,
    ensure_config_schema,
    load_key,
    load_key_or,
    task_config_context,
    update_key,
    update_keys,
    yaml,
)
from core.workspace import (
    DEFAULT_GLOSSARY_DOMAIN,
    TaskWorkspace,
    create_glossary_domain,
    create_task_workspace,
    delete_glossary_domain,
    domain_dict_json_path,
    get_glossary_domain_meta,
    list_glossary_domains,
    list_task_workspaces,
    load_domain_glossary,
    load_global_glossary,
    promote_terms_to_domain,
    rmtree_robust,
    sanitize_domain_name,
    save_domain_glossary,
    save_global_glossary,
    workspace_context,
)


def _configure_utf8_console():
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
WEBUI_DIR = Path(__file__).resolve().parent / "core" / "webui"
WEBUI_HTML_PATH = WEBUI_DIR / "index.html"
WEBUI_STATIC_ASSETS = {
    "base.css": "text/css; charset=utf-8",
    "workbench.css": "text/css; charset=utf-8",
    "views.css": "text/css; charset=utf-8",
    "data.js": "application/javascript; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
}

_sse_subscribers: list[queue.Queue[dict[str, Any]]] = []
_sse_lock = Lock()
_shutdown_event = Event()


def _trigger_shutdown() -> None:
    """Signal all active SSE streams and background runner threads to stop immediately."""
    _shutdown_event.set()
    runner.stop()
    with _sse_lock:
        for q in list(_sse_subscribers):
            try:
                q.put_nowait({"type": "shutdown"})
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait({"type": "shutdown"})
                except queue.Full:
                    pass


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
            try:
                ws.log_dir.mkdir(parents=True, exist_ok=True)
                time_str = payload.get("time") or ""
                label_str = payload.get("log_label") or "RUN"
                with open(ws.log_dir / "runtime.log", "a", encoding="utf-8") as log_f:
                    log_f.write(f"{time_str} [{label_str}] {msg}\n")
            except OSError:
                pass
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
    with _sse_lock:
        for q in list(_sse_subscribers):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


runner.add_listener(_on_runner_event)


# ─── Request Models ───


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] | None = None
    dubbing: bool | None = None
    target_language: str | None = Field(default=None, min_length=1)
    source_language: str | None = Field(default=None, min_length=1)
    glossary_domain: str | None = None


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    subtitles: str | None = None
    title: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    glossary_domain: str | None = None
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "subtitles"
    dubbing: bool = False
    autorun: bool = False


class InvalidateStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = Field(min_length=1)
    autorun: bool = False
    run_stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "subtitles"


class TermsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    terms: list[dict[str, Any]] | None = None
    summary: str | None = None
    glossary_domain: str | None = None


class CreateDomainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str | None = None
    name: str | None = None


class UtterancesUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    utterances: list[dict[str, Any]]


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: dict[str, Any]


# ─── Shared Helpers ───


def _clean_user_path(raw: str | Path) -> str:
    """Strip whitespace and surrounding quotes from Windows 'Copy as path' strings or file:/// URIs."""
    s = str(raw or "").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    if s.lower().startswith("file:///"):
        from urllib.parse import unquote

        rest = unquote(s[8:])
        if len(rest) >= 2 and rest[1] in (":", "|"):
            s = rest[0] + ":" + rest[2:]
        else:
            s = "/" + rest
    return s


def subtitle_file(path):
    """The SRT file of a request, which has to be one with subtitles in it."""
    from core._2_import_subtitles import read_cues

    path = Path(_clean_user_path(path)).resolve()
    if (
        not path.is_file()
        or path.suffix.lower() != ".srt"
        or path.is_relative_to(OUTPUT.resolve())
    ):
        raise HTTPException(
            422, "Subtitles must be an existing SRT file outside output/"
        )
    try:
        read_cues(path.read_bytes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return path


def _wrap_workspace_steps(
    ws: TaskWorkspace,
    steps: list[tuple[str, Any]],
    config_override: dict[str, Any] | None = None,
) -> list[tuple[str, Any]]:
    def _run_step(fn, overrides):
        with workspace_context(ws.root), task_config_context(overrides):
            return fn()

    return [(label, partial(_run_step, fn, config_override)) for label, fn in steps]


def _resolve_workspace(task_id: str) -> TaskWorkspace:
    if task_id == "output" and OUTPUT.is_dir():
        return TaskWorkspace(OUTPUT)
    task_dir = (OUTPUT / task_id).resolve()
    if task_dir.parent != OUTPUT.resolve() or not task_dir.is_dir():
        raise HTTPException(404, f"Task workspace not found: {task_id}")
    return TaskWorkspace(task_dir)


def _workspace_summary(ws: TaskWorkspace) -> dict[str, Any]:
    manifest = ws.load_manifest()
    files = sorted(
        p.name for p in ws.root.glob("*") if p.is_file() and not p.name.startswith(".")
    )
    for audio_sub in ("dub.wav", "raw.mp3", "vocal.mp3", "background.mp3"):
        if (ws.audio_dir / audio_sub).is_file():
            files.append(f"audio/{audio_sub}")
    refers = (
        sorted(p.name for p in ws.refers_dir.glob("*.wav") if p.is_file())
        if ws.refers_dir.is_dir()
        else []
    )
    segs = (
        sorted(p.name for p in ws.segs_dir.glob("*.wav") if p.is_file())
        if ws.segs_dir.is_dir()
        else []
    )
    truncated = []
    trunc_path = ws.log_dir / "dub_truncated.json"
    if trunc_path.is_file():
        data = json.loads(trunc_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(
                f"Expected list in {trunc_path}, got {type(data).__name__}"
            )
        truncated = data
    stages_done = ws.compute_stages_done()
    if runner.active_task_id == ws.task_id:
        manifest["execution"] = {
            **manifest.get("execution", {}),
            "state": runner.state,
            "current_step": runner.current_label or None,
            "sub_step": runner.sub_label or None,
            "sub_progress": round(runner.sub_progress, 4),
            "step_index": runner.current_step,
            "total_steps": runner.total_steps,
            "progress": runner.progress,
            "pause_reason": runner.pause_message or None,
            "error": runner.error_msg or None,
        }
    elif runner.is_queued(ws.task_id):
        manifest["execution"] = {
            **manifest.get("execution", {}),
            "state": "queued",
            "current_step": "Queued",
        }
    manifest["files"] = files
    manifest["refers"] = refers
    manifest["segs"] = segs
    manifest["truncated"] = truncated
    manifest["stages_done"] = stages_done
    return manifest


# ─── Web UI & SSE Endpoints ───


@app.get("/", response_class=HTMLResponse)
def webui_index():
    if not WEBUI_HTML_PATH.is_file():
        raise HTTPException(404, "Web UI HTML not found")
    return HTMLResponse(WEBUI_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/events")
async def sse_events(request: Request):
    q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=128)
    with _sse_lock:
        _sse_subscribers.append(q)

    async def event_generator():
        try:
            init_payload = {
                "type": "connected",
                "state": runner.state,
                "active_task_id": runner.active_task_id,
                "progress": runner.progress,
                "stages_done": dict(runner.stages_done) if runner.stages_done else None,
                "recent_logs": runner.recent_logs(),
            }
            yield f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
            while not _shutdown_event.is_set():
                if await request.is_disconnected():
                    break
                try:
                    event = q.get_nowait()
                    if event.get("type") == "shutdown" or _shutdown_event.is_set():
                        break
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    for _ in range(4):
                        if _shutdown_event.is_set() or not q.empty():
                            break
                        await asyncio.sleep(0.1)
                    if _shutdown_event.is_set() or await request.is_disconnected():
                        break
                    if q.empty():
                        yield 'data: {"type": "heartbeat"}\n\n'
        finally:
            with _sse_lock:
                if q in _sse_subscribers:
                    _sse_subscribers.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/dialogs/pick-files")
def pick_local_files():
    """Open native OS file picker for zero-copy local video selection."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        script = (
            'set chosenFiles to choose file with prompt "Select video/audio files for VideoLingo" '
            "with multiple selections allowed\n"
            "set posixPaths to {}\n"
            "repeat with aFile in chosenFiles\n"
            "  set end of posixPaths to POSIX path of aFile\n"
            "end repeat\n"
            'set AppleScript\'s text item delimiters to "\\n"\n'
            "return posixPaths as text"
        )
        proc = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            return {"paths": []}
        paths = [p.strip() for p in (proc.stdout or "").splitlines() if p.strip()]
        return {"paths": paths}
    if sys_name == "Windows":
        ps_script = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$f = New-Object System.Windows.Forms.OpenFileDialog; "
            "$f.Title = '选择视频或音频文件 - VideoLingo'; "
            "$f.Filter = '媒体与字幕文件 (*.mp4;*.mov;*.mkv;*.avi;*.flv;*.wmv;*.webm;*.wav;*.mp3;*.flac;*.m4a;*.srt)|*.mp4;*.mov;*.mkv;*.avi;*.flv;*.wmv;*.webm;*.wav;*.mp3;*.flac;*.m4a;*.srt|所有文件 (*.*)|*.*'; "
            "$f.Multiselect = $false; "
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.TopMost = $true; "
            "$top.ShowInTaskbar = $false; "
            "$top.Width = 0; $top.Height = 0; "
            "$top.WindowState = 'Minimized'; $top.Show(); $top.WindowState = 'Normal'; $top.Activate(); "
            "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }; "
            "$top.Dispose()"
        )
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-STA",
                "-Command",
                ps_script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if proc.returncode != 0:
            return {"paths": []}
        paths = [p.strip() for p in (proc.stdout or "").splitlines() if p.strip()]
        return {"paths": paths}
    return {"paths": []}


# ─── Multi-Task Workspace & Queue Endpoints ───


@app.get("/api/tasks")
def list_tasks():
    workspaces = list_task_workspaces(OUTPUT)
    return {"tasks": [_workspace_summary(ws) for ws in workspaces]}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    ws = _resolve_workspace(task_id)
    return _workspace_summary(ws)


@app.post("/api/tasks", status_code=202)
def create_task(request: CreateTaskRequest):
    source_str = _clean_user_path(request.source)
    parsed = urlparse(source_str)
    overrides: dict[str, Any] = {}
    if request.source_language:
        overrides["whisper.language"] = request.source_language
        if request.source_language != "auto":
            overrides["whisper.detected_language"] = request.source_language
    if request.target_language:
        overrides["target_language"] = request.target_language
    chosen_domain = (
        request.glossary_domain.strip()
        if request.glossary_domain is not None
        else str(load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or "").strip()
    )
    overrides["glossary_domain"] = chosen_domain
    if request.glossary_domain is not None:
        update_key("glossary_domain", chosen_domain, add_missing=True)

    sub_path = subtitle_file(request.subtitles) if request.subtitles else None

    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise HTTPException(422, "Invalid URL")
        ws = create_task_workspace(request.title or "youtube_video", base_dir=OUTPUT)
        needs_dub = request.stage == "dubbing" or (
            request.stage == "all" and request.dubbing
        )
        manifest = ws.load_manifest()
        manifest["title"] = request.title or source_str
        manifest["origin"] = {
            "kind": "youtube",
            "source_path": None,
            "original_filename": None,
            "url": source_str,
            "mount_mode": "yt-dlp",
        }
        if sub_path:
            dst_srt = ws.root / "source.srt"
            shutil.copy2(sub_path, dst_srt)
            manifest.setdefault("media", {})["input_subtitle"] = "source.srt"
        manifest["config_override"] = overrides
        manifest.setdefault("execution", {}).update(
            {
                "desired_stage": request.stage,
                "dubbing_enabled": needs_dub,
            }
        )
        ws.save_manifest(manifest)

        def _dl():
            from core._1_ytdlp import download_video_ytdlp

            download_video_ytdlp(
                source_str,
                save_path=str(ws.root),
                resolution=load_key("ytb_resolution"),
            )

        dl_steps = [("Download video", _dl)]
        if request.autorun:
            dl_steps.extend(get_steps(request.stage, needs_dub))
        runner.enqueue(
            _wrap_workspace_steps(ws, dl_steps, overrides), task_id=ws.task_id
        )
        return _workspace_summary(ws)

    src_path = Path(source_str).resolve()
    if not src_path.is_file():
        raise HTTPException(422, f"Source file does not exist: {src_path}")
    formats = load_key("allowed_video_formats") + load_key("allowed_audio_formats")
    if src_path.suffix.lower() == ".srt":
        src_path = subtitle_file(src_path)
    elif src_path.suffix.lower().lstrip(".") not in formats:
        raise HTTPException(422, "Unsupported media format")

    ws = create_task_workspace(request.title or src_path.stem, base_dir=OUTPUT)
    ws.init_from_local_file(
        src_path,
        subtitle_path=sub_path,
        title=request.title or src_path.stem,
        config_override=overrides,
    )
    needs_dub = request.stage == "dubbing" or (
        request.stage == "all" and request.dubbing
    )
    ws.update_manifest(
        execution={
            "desired_stage": request.stage,
            "dubbing_enabled": needs_dub,
        }
    )
    if request.autorun:
        steps = _wrap_workspace_steps(
            ws, get_steps(request.stage, needs_dub), overrides
        )
        runner.enqueue(steps, task_id=ws.task_id)
    return _workspace_summary(ws)


@app.post("/api/tasks/upload", status_code=202)
def upload_task(
    file: UploadFile = File(...),
    subtitles: UploadFile | None = File(None),
    subtitles_path: str | None = Form(None),
    source_language: str = Form("en"),
    target_language: str = Form("简体中文"),
    glossary_domain: str | None = Form(None),
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] = Form("subtitles"),
    autorun: bool = Form(False),
):
    raw_name = file.filename or "uploaded_video.mp4"
    ext = Path(raw_name).suffix.lower() or ".mp4"
    stem = Path(raw_name).stem or "video"
    formats = load_key("allowed_video_formats") + load_key("allowed_audio_formats")
    if ext != ".srt" and ext.lstrip(".") not in formats:
        raise HTTPException(422, f"Unsupported media format: {ext}")

    payload_bytes: bytes | None = None
    if ext == ".srt":
        from core._2_import_subtitles import read_cues

        payload_bytes = file.file.read()
        try:
            read_cues(payload_bytes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    sub_bytes: bytes | None = None
    sub_local_path: Path | None = None
    if subtitles is not None and subtitles.filename:
        from core._2_import_subtitles import read_cues

        sub_bytes = subtitles.file.read()
        try:
            read_cues(sub_bytes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    elif subtitles_path and subtitles_path.strip():
        sub_local_path = subtitle_file(subtitles_path.strip())

    ws = create_task_workspace(stem, base_dir=OUTPUT)
    dst_name = "source.srt" if ext == ".srt" else f"source{ext}"
    dst_path = ws.root / dst_name
    if payload_bytes is not None:
        dst_path.write_bytes(payload_bytes)
    else:
        with open(dst_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

    chosen_domain = (
        glossary_domain.strip()
        if glossary_domain is not None
        else str(load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or "").strip()
    )
    if glossary_domain is not None:
        update_key("glossary_domain", chosen_domain, add_missing=True)

    overrides = {
        "whisper.language": source_language,
        "target_language": target_language,
        "glossary_domain": chosen_domain,
    }
    if source_language != "auto":
        overrides["whisper.detected_language"] = source_language
    ws.init_from_local_file(
        dst_path,
        subtitle_path=sub_local_path,
        title=stem,
        config_override=overrides,
    )
    if sub_bytes is not None:
        (ws.root / "source.srt").write_bytes(sub_bytes)
        ws.update_manifest(
            subtitles={
                "source_type": "uploaded_srt",
                "original_path": subtitles.filename if subtitles else "source.srt",
                "filename": "source.srt",
            }
        )
    needs_dub = stage in ("all", "dubbing")
    ws.update_manifest(
        execution={
            "desired_stage": stage,
            "dubbing_enabled": needs_dub,
        }
    )
    if autorun:
        steps = _wrap_workspace_steps(ws, get_steps(stage, needs_dub), overrides)
        runner.enqueue(steps, task_id=ws.task_id)
    return _workspace_summary(ws)


@app.post("/api/tasks/{task_id}/run", status_code=202)
def run_task(task_id: str, request: RunRequest):
    ws = _resolve_workspace(task_id)
    if (runner.active_task_id == ws.task_id and runner.is_active) or runner.is_queued(
        ws.task_id
    ):
        raise HTTPException(409, f"Task {ws.task_id} is already running or queued.")
    manifest = ws.load_manifest()
    exec_info = manifest.get("execution") or {}
    saved_stage = exec_info.get("desired_stage")
    effective_stage: Literal["transcribe", "subtitles", "dubbing", "all"] = (
        request.stage
        if request.stage is not None
        else (
            saved_stage
            if saved_stage in ("transcribe", "subtitles", "dubbing", "all")
            else "subtitles"
        )
    )
    effective_dub = (
        request.dubbing
        if request.dubbing is not None
        else bool(exec_info.get("dubbing_enabled", False))
    )

    media_info = manifest.get("media") or {}
    media_kind = media_info.get("kind") or media_info.get("type")
    origin_url = (manifest.get("origin") or {}).get("url")
    needs_download = False
    try:
        _, resolved_kind = ws.resolve_media_file()
        media_kind = resolved_kind
    except ValueError as exc:
        if origin_url:
            needs_download = True
            media_kind = media_kind or "video"
        elif not media_kind:
            raise HTTPException(409, str(exc)) from exc

    needs_dubbing = effective_stage == "dubbing" or (
        effective_stage == "all" and effective_dub
    )
    if needs_dubbing and media_kind in ("audio", "subtitle"):
        raise HTTPException(
            422, "Input without a video supports subtitles only, as in the UI."
        )
    if effective_stage == "dubbing" and not (
        ws.has_utterances()
        or all((ws.root / name).exists() for name in ("src.srt", "trans.srt"))
    ):
        raise HTTPException(409, "Generate subtitles before starting dubbing.")

    overrides = dict(manifest.get("config_override") or {})
    if request.target_language:
        overrides["target_language"] = request.target_language
    if request.source_language:
        overrides["whisper.language"] = request.source_language
        if request.source_language != "auto":
            overrides["whisper.detected_language"] = request.source_language
    if request.glossary_domain is not None:
        overrides["glossary_domain"] = request.glossary_domain.strip()
    manifest["config_override"] = overrides
    manifest.setdefault("execution", {}).update(
        {
            "desired_stage": effective_stage,
            "dubbing_enabled": needs_dubbing,
        }
    )
    ws.save_manifest(manifest)

    raw_steps = list(get_steps(effective_stage, needs_dubbing))
    if needs_download and origin_url:

        def _dl():
            from core._1_ytdlp import download_video_ytdlp

            download_video_ytdlp(
                origin_url,
                save_path=str(ws.root),
                resolution=load_key("ytb_resolution"),
            )

        raw_steps = [("Download video", _dl)] + raw_steps

    steps = _wrap_workspace_steps(ws, raw_steps, overrides)
    started_now = runner.enqueue(steps, task_id=ws.task_id)
    return {"accepted": True, "started_immediately": started_now, "task_id": ws.task_id}


@app.post("/api/tasks/{task_id}/pause")
def pause_task(task_id: str):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id != ws.task_id or not runner.is_active:
        raise HTTPException(409, f"Task {ws.task_id} is not the active running task.")
    runner.pause("Paused by user")
    return {"state": runner.state}


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id != ws.task_id or runner.state != "paused":
        raise HTTPException(409, f"Task {ws.task_id} is not paused.")
    runner.resume()
    return {"state": runner.state}


@app.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
        runner.stop(clear_queue=False)
        return {"state": runner.state}
    if runner.cancel_queued(ws.task_id):
        ws.update_manifest(execution={"state": "stopped"})
        return {"state": "stopped"}
    raise HTTPException(409, f"Task {ws.task_id} is not active.")


@app.post("/api/tasks/{task_id}/invalidate")
def invalidate_task_stage(task_id: str, request: InvalidateStageRequest):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
        if runner.state == "paused":
            runner.stop()
            if runner._thread is not None and runner._thread.is_alive():
                runner._thread.join(timeout=2.0)
        if runner.is_active:
            raise HTTPException(
                409,
                "Cannot invalidate stages while the task is actively running; stop it first.",
            )
    try:
        invalidated = ws.invalidate_from_stage(request.stage)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if request.autorun:
        overrides = dict(ws.load_manifest().get("config_override") or {})
        needs_dub = request.run_stage in ("dubbing", "all")
        steps = _wrap_workspace_steps(
            ws, get_steps(request.run_stage, needs_dub), overrides
        )
        runner.enqueue(steps, task_id=ws.task_id)
    return {"invalidated": invalidated}


@app.get("/api/tasks/{task_id}/terms")
def get_task_terms(task_id: str):
    ws = _resolve_workspace(task_id)
    if ws.has_terms():
        return ws.load_terms()
    return {"summary": "", "terms": []}


@app.put("/api/tasks/{task_id}/terms")
def update_task_terms(task_id: str, request: TermsUpdateRequest):
    ws = _resolve_workspace(task_id)
    manifest = ws.load_manifest()
    overrides = dict(manifest.get("config_override") or {})
    if request.glossary_domain is not None:
        overrides["glossary_domain"] = request.glossary_domain.strip()
        manifest["config_override"] = overrides
        ws.save_manifest(manifest)

    task_domain = overrides.get("glossary_domain")
    if task_domain is None:
        task_domain = str(
            load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or ""
        ).strip()
    else:
        task_domain = str(task_domain).strip()

    if request.terms is None:
        current = get_task_terms(task_id)
        return {
            "saved": len(current.get("terms") or []),
            "promoted_to_domain": 0,
            "promoted_to_global": 0,
            "glossary_domain": task_domain,
        }

    current = get_task_terms(task_id)
    summary = (
        request.summary if request.summary is not None else current.get("summary", "")
    )
    to_promote = [
        t
        for t in request.terms
        if t.get("promote_to_domain") or t.get("promote_to_global")
    ]
    promoted = (
        promote_terms_to_domain(to_promote, domain=task_domain)
        if (to_promote and task_domain)
        else 0
    )
    ws.save_terms(summary, request.terms)
    loaded = ws.load_terms()
    return {
        "saved": len(loaded["terms"]),
        "promoted_to_domain": promoted,
        "promoted_to_global": promoted,
        "glossary_domain": task_domain,
    }


@app.get("/api/tasks/{task_id}/utterances")
def get_task_utterances(task_id: str):
    ws = _resolve_workspace(task_id)
    return {"utterances": ws.load_utterances()}


@app.put("/api/tasks/{task_id}/utterances")
def update_task_utterances(task_id: str, request: UtterancesUpdateRequest):
    ws = _resolve_workspace(task_id)
    existing_utts = ws.load_utterances()
    normalized_utts = []
    for idx, u in enumerate(request.utterances):
        item = dict(u)
        orig_u = existing_utts[idx] if idx < len(existing_utts) else {}
        if "src_punct" not in item and orig_u.get("src_punct"):
            item["src_punct"] = orig_u["src_punct"]
        src = str(item.get("src") or "").strip()
        tgt = str(item.get("tgt") or "").strip()
        tts_text = str(item.get("tts_text") or tgt).strip()
        item["src"] = src
        if src != str(orig_u.get("src") or "").strip() or not item.get("src_punct"):
            item["src_punct"] = src
        item["tgt"] = tgt
        item["tts_text"] = tts_text
        cues = item.get("sub_cues")
        default_cue = {
            "src": src,
            "tgt": tgt,
            "trans": tgt,
            "word_span": list(item.get("word_span") or [0, 0]),
            "start": float(item.get("start") or 0.0),
            "end": float(item.get("end") or 0.0),
        }
        if isinstance(cues, list) and len(cues) == 1:
            c0 = dict(cues[0]) if isinstance(cues[0], dict) else {}
            item["sub_cues"] = [
                {**default_cue, **c0, "src": src, "tgt": tgt, "trans": tgt}
            ]
        elif isinstance(cues, list) and len(cues) > 1:
            prev_src = " ".join(
                str(c.get("src") or "").strip() for c in cues if c.get("src")
            ).strip()
            prev_tgt = " ".join(
                str(c.get("tgt", c.get("trans", c.get("text", ""))) or "").strip()
                for c in cues
                if c.get("tgt") or c.get("trans") or c.get("text")
            ).strip()
            if src != prev_src or tgt != prev_tgt:
                item["sub_cues"] = [default_cue]
        normalized_utts.append(item)
    ws.save_utterances(normalized_utts)
    ws.delete_kv("sub_split_cfg", "sub_split_rev")
    if ws.has_dub_tasks():
        ws.invalidate_from_stage("dub_tts")
    is_task_active = runner.active_task_id == ws.task_id and runner.is_active
    if (
        normalized_utts
        and not is_task_active
        and ws.has_words()
        and (ws.src_srt_path.is_file() or ws.trans_srt_path.is_file())
    ):
        from core._6_gen_sub import align_timestamp_main

        manifest = ws.load_manifest()
        overrides = dict(manifest.get("config_override") or {})
        with workspace_context(ws), task_config_context(overrides):
            try:
                align_timestamp_main()
            except Exception as exc:
                raise HTTPException(
                    500, f"Failed to realign subtitle files: {exc}"
                ) from exc
    return {"saved": len(normalized_utts)}


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
        raise HTTPException(409, "Cannot delete an active task; stop it first.")
    runner.cancel_queued(ws.task_id)
    ws.clear_runtime_cache()
    rmtree_robust(ws.root)
    return {"deleted": ws.task_id}


@app.get("/api/tasks/{task_id}/files/{name:path}")
def get_task_file(task_id: str, name: str):
    ws = _resolve_workspace(task_id)
    root_resolved = ws.root.resolve()
    path = (ws.root / name).resolve()
    if (
        not path.is_relative_to(root_resolved)
        or path == root_resolved
        or any(part.startswith(".") for part in path.relative_to(root_resolved).parts)
        or not path.is_file()
    ):
        raise HTTPException(404, "Task file not found")
    title = ws.load_manifest().get("title") or ws.task_id
    safe_title = (
        "".join(c for c in str(title) if c not in '<>:"/\\|?*').strip() or "video"
    )
    dl_name = (
        f"{safe_title}_{path.name}" if path.name.startswith("output_") else path.name
    )
    return FileResponse(path, filename=dl_name)


class LLMProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


class DetectMaiRegionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=1)


@app.post("/api/mai/detect-region")
def detect_mai_region(request: DetectMaiRegionRequest):
    from core.asr_backend.mai_asr import detect_region

    region = detect_region(request.api_key.strip())
    if region:
        update_key("whisper.mai_region", region, add_missing=True)
    return {"region": region}


@app.get("/api/tasks/{task_id}/subtitles.zip")
def download_task_subtitles_zip(task_id: str):
    ws = _resolve_workspace(task_id)
    sub_files = [
        p
        for p in sorted(ws.root.glob("*"))
        if p.is_file() and p.suffix.lower() in {".srt", ".ass", ".vtt"}
    ]
    if not sub_files:
        raise HTTPException(404, "No subtitle files generated yet")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sub_files:
            zf.write(p, arcname=p.name)
    title = ws.load_manifest().get("title") or ws.task_id
    safe_title = (
        "".join(c for c in str(title) if c not in '<>:"/\\|?*').strip() or "subtitles"
    )
    ascii_fallback = (
        "".join(
            c for c in safe_title if c.isascii() and (c.isalnum() or c in "-_.")
        ).strip()
        or ws.task_id
    )
    encoded_utf8 = quote(f"{safe_title}_subtitles.zip")
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_fallback}_subtitles.zip"; '
                f"filename*=UTF-8''{encoded_utf8}"
            )
        },
    )


@app.post("/api/llm/models")
def fetch_llm_models(request: LLMProbeRequest):
    import requests
    from core.utils.ask_gpt import is_local_endpoint, normalize_base_url

    base_url = (request.base_url or str(load_key("api.base_url") or "")).strip()
    api_key = (
        request.api_key
        if request.api_key is not None
        else str(load_key("api.key") or "")
    ).strip()
    if not base_url or (not api_key and not is_local_endpoint(base_url)):
        return {"models": [], "error": "Missing API Base URL or API Key"}
    url = normalize_base_url(base_url) + "/models"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = sorted(str(m["id"]) for m in data if isinstance(m, dict) and "id" in m)
        return {"models": models}
    except Exception as exc:
        return {"models": [], "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


@app.post("/api/llm/check")
def check_llm_api(request: LLMProbeRequest):
    from core.utils.ask_gpt import ask_gpt

    overrides: dict[str, Any] = {}
    if request.base_url is not None and request.base_url.strip():
        overrides["api.base_url"] = request.base_url.strip()
    if request.api_key is not None and request.api_key.strip():
        overrides["api.key"] = request.api_key.strip()
    if request.model is not None and request.model.strip():
        overrides["api.model"] = request.model.strip()
    if request.reasoning_effort is not None and request.reasoning_effort.strip():
        overrides["api.reasoning_effort"] = request.reasoning_effort.strip()

    try:
        fn = getattr(ask_gpt, "__wrapped__", ask_gpt)
        with task_config_context(overrides):
            resp = fn(
                f"This is a connectivity test ({int(time.time())}), respond with {{'message':'success'}} in JSON format.",
                resp_type="json",
                log_title="None",
            )
        if isinstance(resp, dict) and resp.get("message") == "success":
            return {"valid": True, "error": ""}
        return {"valid": False, "error": f"Unexpected response: {str(resp)[:240]}"}
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}


@app.get("/api/glossary")
def get_global_glossary(domain: str | None = None):
    if domain is not None:
        d = domain.strip()
        if not d:
            return {"domain": "", "terms": [], "has_dict": False, "dict_count": 0}
        try:
            meta = get_glossary_domain_meta(d)
            terms = load_domain_glossary(d)
            return {
                "domain": meta["name"],
                "terms": terms,
                "has_dict": meta["has_dict"],
                "dict_count": meta["dict_count"],
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"terms": load_global_glossary()}


@app.put("/api/glossary")
def put_global_glossary(request: TermsUpdateRequest, domain: str | None = None):
    terms_list = request.terms or []
    target_domain = (
        domain
        if domain is not None
        else (request.glossary_domain if request.glossary_domain is not None else None)
    )
    if target_domain is not None:
        d = target_domain.strip()
        if not d:
            return {"saved": 0, "domain": ""}
        try:
            save_domain_glossary(d, terms_list)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"saved": len(terms_list), "domain": d}
    save_global_glossary(terms_list)
    return {"saved": len(terms_list)}


@app.get("/api/glossary/domains")
def api_list_glossary_domains():
    domains = list_glossary_domains()
    active = str(load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or "").strip()
    return {"domains": domains, "active_domain": active}


@app.post("/api/glossary/domains", status_code=201)
def api_create_glossary_domain(request: CreateDomainRequest):
    raw_name = (request.domain or request.name or "").strip()
    if not raw_name:
        raise HTTPException(422, "Domain name is required")
    try:
        meta = create_glossary_domain(raw_name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    update_key("glossary_domain", meta["name"], add_missing=True)
    return {
        "domain": meta["name"],
        "meta": meta,
        "domains": list_glossary_domains(),
    }


@app.delete("/api/glossary/domains/{domain}")
def api_delete_glossary_domain(domain: str):
    try:
        safe = sanitize_domain_name(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    deleted = delete_glossary_domain(safe)
    if not deleted:
        raise HTTPException(404, f"Glossary domain not found: {safe}")
    active = str(load_key_or("glossary_domain", DEFAULT_GLOSSARY_DOMAIN) or "").strip()
    if active == safe:
        update_key("glossary_domain", "", add_missing=True)
        active = ""
    return {
        "deleted": safe,
        "active_domain": active,
        "domains": list_glossary_domains(),
    }


@app.get("/api/glossary/domains/{domain}")
def api_get_glossary_domain(domain: str):
    try:
        meta = get_glossary_domain_meta(domain)
        terms = load_domain_glossary(meta["name"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**meta, "terms": terms}


@app.put("/api/glossary/domains/{domain}")
def api_put_glossary_domain(domain: str, request: TermsUpdateRequest):
    try:
        safe = sanitize_domain_name(domain)
        save_domain_glossary(safe, request.terms or [])
        meta = get_glossary_domain_meta(safe)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"saved": meta["terms_count"], "domain": meta["name"], "meta": meta}


@app.get("/api/glossary/domains/{domain}/dict")
def api_search_glossary_domain_dict(domain: str, q: str = "", limit: int = 50):
    try:
        dict_file = domain_dict_json_path(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not dict_file.is_file():
        return {"entries": None, "total": 0}
    try:
        parsed = json.loads(dict_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise HTTPException(
            500, f"Failed to read domain dictionary {dict_file.name}: {exc}"
        ) from exc
    pairs: list[dict[str, str]] = []
    if isinstance(parsed, dict):
        terms_list = parsed.get("terms")
        if isinstance(terms_list, list):
            for item in terms_list:
                if isinstance(item, dict) and item.get("src") and item.get("tgt"):
                    pairs.append({"k": str(item["src"]), "v": str(item["tgt"])})
        else:
            for k_val, v_val in parsed.items():
                if isinstance(v_val, str):
                    pairs.append({"k": str(k_val), "v": v_val})
    query = q.strip().lower()
    matched = [
        p
        for p in pairs
        if not query or query in p["k"].lower() or query in p["v"].lower()
    ]
    return {"entries": matched[: max(1, min(limit, 200))], "total": len(matched)}


@app.delete("/api/glossary/domains/{domain}/dict")
def api_delete_glossary_domain_dict(domain: str):
    try:
        dict_file = domain_dict_json_path(domain)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if dict_file.is_file():
        dict_file.unlink()
    meta = get_glossary_domain_meta(domain)
    return {
        "domain": meta["name"],
        "meta": meta,
        "terms": load_domain_glossary(meta["name"]),
    }


@app.post("/api/glossary/domains/{domain}/import")
def api_import_glossary_domain_file(domain: str, file: UploadFile = File(...)):
    import pandas as pd
    from core._4_1_summarize import _extract_generic_kb_pairs, _normalize_aliases

    try:
        safe = sanitize_domain_name(domain)
        create_glossary_domain(safe)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    raw_name = file.filename or "terms.csv"
    ext = Path(raw_name).suffix.lower()
    content = file.file.read()
    if not content:
        raise HTTPException(422, "Uploaded file is empty")

    if ext == ".json":
        try:
            parsed = json.loads(content.decode("utf-8-sig"))
        except Exception as exc:
            raise HTTPException(422, f"Invalid JSON file: {exc}") from exc
        extracted: dict[str, dict[str, Any]] = {}
        _extract_generic_kb_pairs(parsed, extracted, source_tag="domain_dict")
        if not extracted:
            raise HTTPException(422, "No valid bilingual term pairs found in JSON")
        dict_file = domain_dict_json_path(safe)
        dict_file.parent.mkdir(parents=True, exist_ok=True)
        dict_file.write_bytes(content)
        meta = get_glossary_domain_meta(safe)
        meta["dict_count"] = len(extracted)
        return {
            "imported_as": "dict.json",
            "imported_kind": "dict",
            "imported_count": len(extracted),
            "dict_count": len(extracted),
            "terms_count": meta["terms_count"],
            "domain": meta["name"],
            "meta": meta,
            "terms": load_domain_glossary(safe),
        }

    def _read_csv_with_encoding_fallback(
        raw_bytes: bytes, **kwargs: Any
    ) -> pd.DataFrame:
        for enc in ("utf-8-sig", "gb18030", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig", **kwargs)

    if ext in {".csv", ".xlsx", ".xls"}:
        try:
            if ext == ".csv":
                df = _read_csv_with_encoding_fallback(content)
            else:
                df = pd.read_excel(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(422, f"Could not parse {ext} file: {exc}") from exc

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        has_header = any(
            k in cols_lower
            for k in ("src", "source", "original", "term", "en", "原文", "原词")
        )
        if not has_header:
            if ext == ".csv":
                df = _read_csv_with_encoding_fallback(
                    content,
                    header=None,
                    names=["src", "tgt", "note", "aliases", "fix_source"],
                )
            else:
                df = pd.read_excel(io.BytesIO(content), header=None)
                col_names = ["src", "tgt", "note", "aliases", "fix_source"]
                df.columns = [
                    col_names[i] if i < len(col_names) else f"col_{i}"
                    for i in range(len(df.columns))
                ]
            cols_lower = {str(c).strip().lower(): c for c in df.columns}

        src_col = (
            cols_lower.get("src")
            or cols_lower.get("source")
            or cols_lower.get("original")
            or cols_lower.get("term")
            or cols_lower.get("en")
            or cols_lower.get("原文")
            or cols_lower.get("原词")
            or (df.columns[0] if len(df.columns) >= 1 else None)
        )
        tgt_col = (
            cols_lower.get("tgt")
            or cols_lower.get("target")
            or cols_lower.get("translation")
            or cols_lower.get("zh")
            or cols_lower.get("zh-chs")
            or cols_lower.get("译文")
            or cols_lower.get("译名")
            or (df.columns[1] if len(df.columns) >= 2 else None)
        )
        note_col = (
            cols_lower.get("note")
            or cols_lower.get("notes")
            or cols_lower.get("备注")
            or (df.columns[2] if len(df.columns) >= 3 and not has_header else None)
        )
        aliases_col = (
            cols_lower.get("aliases")
            or cols_lower.get("alias")
            or cols_lower.get("别名")
        )
        fix_col = cols_lower.get("fix_source")

        if src_col is None or tgt_col is None:
            raise HTTPException(
                422, "Table must contain at least source and target columns"
            )

        rows_to_merge: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            s_val = row.get(src_col)
            t_val = row.get(tgt_col)
            if (
                s_val is None
                or t_val is None
                or bool(pd.isna(s_val))
                or bool(pd.isna(t_val))
            ):
                continue
            s = str(s_val).strip()
            t = str(t_val).strip()
            if (
                not s
                or not t
                or s.lower() in {"src", "source", "original", "term", "nan"}
            ):
                continue
            n_val = row.get(note_col) if note_col is not None else ""
            n = "" if (n_val is None or bool(pd.isna(n_val))) else str(n_val).strip()
            al_val = row.get(aliases_col) if aliases_col is not None else ""
            al_raw = (
                "" if (al_val is None or bool(pd.isna(al_val))) else str(al_val).strip()
            )
            f_val = row.get(fix_col) if fix_col is not None else False
            fix_bool = (
                False
                if (f_val is None or bool(pd.isna(f_val)))
                else str(f_val).strip().lower() in {"1", "true", "yes"}
            )
            rows_to_merge.append(
                {
                    "src": s,
                    "tgt": t,
                    "aliases": _normalize_aliases(al_raw, s),
                    "note": n,
                    "fix_source": fix_bool,
                }
            )

        if len(rows_to_merge) > 500:
            has_rich_fields = any(
                item.get("note") or item.get("aliases") or item.get("fix_source")
                for item in rows_to_merge
            )
            dict_payload: Any = (
                {"terms": rows_to_merge}
                if has_rich_fields
                else {item["src"]: item["tgt"] for item in rows_to_merge}
            )
            dict_file = domain_dict_json_path(safe)
            dict_file.parent.mkdir(parents=True, exist_ok=True)
            dict_file.write_text(
                json.dumps(dict_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            meta = get_glossary_domain_meta(safe)
            return {
                "imported_as": "dict.json",
                "imported_kind": "dict",
                "imported_count": len(rows_to_merge),
                "dict_count": len(rows_to_merge),
                "terms_count": meta["terms_count"],
                "domain": meta["name"],
                "meta": meta,
                "terms": load_domain_glossary(safe),
            }

        promoted = promote_terms_to_domain(rows_to_merge, domain=safe)
        meta = get_glossary_domain_meta(safe)
        return {
            "imported_as": "terms.csv",
            "imported_kind": "terms",
            "imported_count": promoted,
            "terms_count": meta["terms_count"],
            "dict_count": meta["dict_count"],
            "domain": meta["name"],
            "meta": meta,
            "terms": load_domain_glossary(safe),
        }

    raise HTTPException(
        422, f"Unsupported glossary file format: {ext} (expected .csv, .xlsx, or .json)"
    )


@app.get("/api/config")
def get_config():
    ensure_config_schema(CONFIG_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return dict(data or {})


@app.put("/api/config")
def put_config(request: ConfigUpdateRequest):
    update_keys(request.updates, add_missing=True)
    return {"updated": list(request.updates.keys())}


@app.post("/api/tasks/{task_id}/open-dir")
def open_task_dir(task_id: str):
    ws = _resolve_workspace(task_id)
    abs_path = str(ws.root.resolve())
    if "PYTEST_CURRENT_TEST" not in os.environ:
        import subprocess
        import sys

        if sys.platform == "darwin":
            subprocess.Popen(["open", abs_path])
        elif sys.platform.startswith("win"):
            os.startfile(abs_path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", abs_path])
    return {"task_id": ws.task_id, "path": abs_path}


@app.get("/{asset_name}")
def webui_static_asset(asset_name: str):
    media_type = WEBUI_STATIC_ASSETS.get(asset_name)
    if media_type is None:
        raise HTTPException(404, "Not found")
    asset_path = WEBUI_DIR / asset_name
    if not asset_path.is_file():
        raise HTTPException(404, f"Web UI asset not found: {asset_name}")
    return Response(
        content=asset_path.read_text(encoding="utf-8"), media_type=media_type
    )


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


if __name__ == "__main__":
    port = int(os.environ.get("VIDEOLINGO_API_PORT", "8501"))
    _create_uvicorn_server(host="127.0.0.1", port=port).run()

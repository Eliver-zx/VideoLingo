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
from threading import Lock
import time
from typing import Any, Literal
from urllib.parse import quote, urlparse
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.pipeline import get_steps
from core.task_runner import TaskRunner
from core.utils.config_utils import (
    CONFIG_PATH,
    ensure_config_schema,
    load_key,
    task_config_context,
    update_key,
    update_keys,
    yaml,
)
from core.workspace import (
    TaskWorkspace,
    create_task_workspace,
    list_task_workspaces,
    load_global_glossary,
    promote_terms_to_global,
    save_global_glossary,
    workspace_context,
)


def _configure_utf8_console():
    """Allow Rich and task threads to print Unicode on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_console()

app = FastAPI(
    title="VideoLingo",
    description="Serial multi-task workspace with SSE and zero-copy local media import.",
)
runner = TaskRunner()
operation_lock = Lock()
OUTPUT = Path("output")
WEBUI_HTML_PATH = Path(__file__).resolve().parent / "core" / "webui" / "index.html"

_sse_subscribers: list[queue.Queue[dict[str, Any]]] = []
_sse_lock = Lock()


def _on_runner_event(payload: dict[str, Any]) -> None:
    task_id = payload.get("task_id")
    if task_id:
        ws = TaskWorkspace(OUTPUT / task_id) if (OUTPUT / task_id).is_dir() else None
        if ws and ws.manifest_path.is_file():
            try:
                ws.update_manifest(
                    execution={
                        "state": payload.get("state"),
                        "current_step": payload.get("step"),
                        "step_index": payload.get("step_index", -1),
                        "total_steps": payload.get("total_steps", 0),
                        "progress": payload.get("progress", 0.0),
                        "pause_reason": payload.get("pause_message"),
                        "error": payload.get("error"),
                    }
                )
            except Exception:
                pass
    with _sse_lock:
        for q in list(_sse_subscribers):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


runner.add_listener(_on_runner_event)


# ─── Request Models ───


class InputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(
        min_length=1,
        description="Local file path or HTTP(S) video URL; an SRT file to translate subtitles without a video",
    )
    subtitles: str | None = Field(
        default=None,
        min_length=1,
        description="Local SRT file that is used instead of the recognition",
    )
    existing: Literal["reject", "archive", "replace"] = "reject"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "all"
    dubbing: bool = False
    target_language: str | None = Field(default=None, min_length=1)
    source_language: str | None = Field(default=None, min_length=1)


class CreateTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    subtitles: str | None = None
    title: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "subtitles"
    dubbing: bool = False
    autorun: bool = True


class InvalidateStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = Field(min_length=1)
    autorun: bool = False
    run_stage: Literal["transcribe", "subtitles", "dubbing", "all"] = "subtitles"


class TermsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    terms: list[dict[str, Any]]
    summary: str | None = None


class UtterancesUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    utterances: list[dict[str, Any]]


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    updates: dict[str, Any]


# ─── Shared Helpers ───


def require_idle():
    if runner.is_active:
        raise HTTPException(
            409, "An operation is active; wait for /status before continuing."
        )


def archive_output():
    from core.utils.onekeycleanup import cleanup

    cleanup()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(
            "Some output files could not be archived; inspect output/ before retrying."
        )


def subtitle_file(path):
    """The SRT file of a request, which has to be one with subtitles in it."""
    from core._2_import_subtitles import read_cues

    path = Path(path).resolve()
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
    stages_done = {
        "asr": ws.words_path.is_file()
        or (ws.log_dir / "cleaned_chunks.xlsx").is_file(),
        "split": ws.utterances_path.is_file()
        or (ws.log_dir / "split_by_meaning.txt").is_file(),
        "terms": ws.terms_path.is_file() or (ws.log_dir / "terminology.json").is_file(),
        "translate": (ws.log_dir / "translation_results.xlsx").is_file(),
        "subtitles": (ws.root / "trans.srt").is_file()
        and (ws.root / "src.srt").is_file(),
        "sub_video": (ws.root / "output_sub.mp4").is_file(),
        "dub_audio": (ws.audio_dir / "dub.wav").is_file()
        or (ws.root / "dub.mp3").is_file(),
        "dub_video": (ws.root / "output_dub.mp4").is_file(),
    }
    if runner.active_task_id == ws.task_id:
        manifest["execution"] = {
            **manifest.get("execution", {}),
            "state": runner.state,
            "current_step": runner.current_label or None,
            "step_index": runner.current_step,
            "total_steps": runner.total_steps,
            "progress": runner.progress,
            "pause_reason": runner.pause_message or None,
            "error": runner.error_msg or None,
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
async def sse_events():
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
            }
            yield f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n"
            while True:
                try:
                    event = q.get_nowait()
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.4)
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
    if platform.system() == "Darwin":
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
    parsed = urlparse(request.source)
    overrides: dict[str, Any] = {}
    if request.source_language:
        overrides["whisper.language"] = request.source_language
        if request.source_language != "auto":
            overrides["whisper.detected_language"] = request.source_language
    if request.target_language:
        overrides["target_language"] = request.target_language

    sub_path = subtitle_file(request.subtitles) if request.subtitles else None

    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            raise HTTPException(422, "Invalid URL")
        ws = create_task_workspace(request.title or "youtube_video", base_dir=OUTPUT)
        manifest = ws.load_manifest()
        manifest["title"] = request.title or request.source
        manifest["origin"] = {
            "kind": "youtube",
            "source_path": None,
            "original_filename": None,
            "url": request.source,
            "mount_mode": "yt-dlp",
        }
        manifest["config_override"] = overrides
        ws.save_manifest(manifest)

        def _dl():
            from core._1_ytdlp import download_video_ytdlp

            download_video_ytdlp(
                request.source,
                save_path=str(ws.root),
                resolution=load_key("ytb_resolution"),
            )

        dl_steps = [("Download video", _dl)]
        if request.autorun:
            needs_dub = request.stage == "dubbing" or (
                request.stage == "all" and request.dubbing
            )
            dl_steps.extend(get_steps(request.stage, needs_dub))
        runner.enqueue(
            _wrap_workspace_steps(ws, dl_steps, overrides), task_id=ws.task_id
        )
        return _workspace_summary(ws)

    src_path = Path(request.source).resolve()
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
    if request.autorun:
        needs_dub = request.stage == "dubbing" or (
            request.stage == "all" and request.dubbing
        )
        steps = _wrap_workspace_steps(
            ws, get_steps(request.stage, needs_dub), overrides
        )
        runner.enqueue(steps, task_id=ws.task_id)
    return _workspace_summary(ws)


@app.post("/api/tasks/upload", status_code=202)
def upload_task(
    file: UploadFile = File(...),
    source_language: str = Form("en"),
    target_language: str = Form("简体中文"),
    stage: Literal["transcribe", "subtitles", "dubbing", "all"] = Form("subtitles"),
    autorun: bool = Form(True),
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

    ws = create_task_workspace(stem, base_dir=OUTPUT)
    dst_name = "source.srt" if ext == ".srt" else f"source{ext}"
    dst_path = ws.root / dst_name
    if payload_bytes is not None:
        dst_path.write_bytes(payload_bytes)
    else:
        with open(dst_path, "wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

    overrides = {
        "whisper.language": source_language,
        "target_language": target_language,
    }
    if source_language != "auto":
        overrides["whisper.detected_language"] = source_language
    ws.init_from_local_file(dst_path, title=stem, config_override=overrides)
    if autorun:
        needs_dub = stage in ("all", "dubbing")
        steps = _wrap_workspace_steps(ws, get_steps(stage, needs_dub), overrides)
        runner.enqueue(steps, task_id=ws.task_id)
    return _workspace_summary(ws)


@app.post("/api/tasks/{task_id}/run", status_code=202)
def run_task(task_id: str, request: RunRequest):
    ws = _resolve_workspace(task_id)
    manifest = ws.load_manifest()
    overrides = dict(manifest.get("config_override") or {})
    if request.target_language:
        overrides["target_language"] = request.target_language
    if request.source_language:
        overrides["whisper.language"] = request.source_language
        if request.source_language != "auto":
            overrides["whisper.detected_language"] = request.source_language
    manifest["config_override"] = overrides
    ws.save_manifest(manifest)

    needs_dubbing = request.stage == "dubbing" or (
        request.stage == "all" and request.dubbing
    )
    steps = _wrap_workspace_steps(
        ws, get_steps(request.stage, needs_dubbing), overrides
    )
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
    if runner.active_task_id != ws.task_id or not runner.is_active:
        raise HTTPException(409, f"Task {ws.task_id} is not active.")
    runner.stop()
    return {"state": runner.state}


@app.post("/api/tasks/{task_id}/invalidate")
def invalidate_task_stage(task_id: str, request: InvalidateStageRequest):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
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
    if ws.terms_path.is_file():
        return ws.load_terms()
    legacy_path = ws.log_dir / "terminology.json"
    if legacy_path.is_file():
        data = json.loads(legacy_path.read_text(encoding="utf-8"))
        return {"summary": data.get("theme", ""), "terms": data.get("terms", [])}
    return {"summary": "", "terms": []}


@app.put("/api/tasks/{task_id}/terms")
def update_task_terms(task_id: str, request: TermsUpdateRequest):
    ws = _resolve_workspace(task_id)
    current = get_task_terms(task_id)
    summary = (
        request.summary if request.summary is not None else current.get("summary", "")
    )
    ws.save_terms(summary, request.terms)

    # Also keep legacy terminology.json synchronized for pipeline compatibility
    loaded = ws.load_terms()
    legacy_payload = {
        "theme": loaded["summary"],
        "terms": [
            {
                "src": t["src"],
                "tgt": t["tgt"],
                "note": t.get("note", ""),
                "aliases": t.get("aliases", []),
                "fix_source": t.get("fix_source", False),
            }
            for t in loaded["terms"]
        ],
    }
    legacy_path = ws.log_dir / "terminology.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(legacy_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    to_promote = [t for t in request.terms if t.get("promote_to_global")]
    promoted = promote_terms_to_global(to_promote) if to_promote else 0
    return {"saved": len(loaded["terms"]), "promoted_to_global": promoted}


@app.get("/api/tasks/{task_id}/utterances")
def get_task_utterances(task_id: str):
    ws = _resolve_workspace(task_id)
    utts = ws.load_utterances()
    if utts:
        return {"utterances": utts}
    trans_xlsx = ws.log_dir / "translation_results.xlsx"
    if trans_xlsx.is_file():
        import pandas as pd

        df = pd.read_excel(trans_xlsx)
        rows = []
        for idx, row in df.iterrows():
            src_val = row.get("Source")
            tgt_val = row.get("Translation")
            src = (
                ""
                if src_val is None or bool(pd.isna(src_val))
                else str(src_val).strip()
            )
            tgt = (
                ""
                if tgt_val is None or bool(pd.isna(tgt_val))
                else str(tgt_val).strip()
            )
            rows.append(
                {
                    "id": idx,
                    "start": 0.0,
                    "end": 0.0,
                    "src": src,
                    "tgt": tgt,
                    "tts_text": tgt,
                }
            )
        return {"utterances": rows}
    return {"utterances": []}


@app.put("/api/tasks/{task_id}/utterances")
def update_task_utterances(task_id: str, request: UtterancesUpdateRequest):
    ws = _resolve_workspace(task_id)
    ws.save_utterances(request.utterances)
    if request.utterances:
        meaning_txt = ws.log_dir / "split_by_meaning.txt"
        if meaning_txt.is_file():
            src_lines = [str(u.get("src") or "").strip() for u in request.utterances]
            meaning_txt.write_text("\n".join(src_lines) + "\n", encoding="utf-8")
        trans_xlsx = ws.log_dir / "translation_results.xlsx"
        if trans_xlsx.is_file():
            import pandas as pd

            df = pd.read_excel(trans_xlsx)
            if len(df) == len(request.utterances):
                df["Source"] = [
                    str(u.get("src") or "").strip() for u in request.utterances
                ]
                df["Translation"] = [
                    str(u.get("tgt") or "").strip() for u in request.utterances
                ]
                df.to_excel(trans_xlsx, index=False)
    return {"saved": len(request.utterances)}


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
        raise HTTPException(409, "Cannot delete an active task; stop it first.")
    shutil.rmtree(ws.root)
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
def get_global_glossary():
    return {"terms": load_global_glossary()}


@app.put("/api/glossary")
def put_global_glossary(request: TermsUpdateRequest):
    save_global_glossary(request.terms)
    return {"saved": len(request.terms)}


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


# ─── Legacy Single-Output Endpoints (Preserved for Existing API Callers & Tests) ───


def prepare_input(source, existing, subtitles=None):
    from core._1_ytdlp import (
        download_video_ytdlp,
        write_input_manifest,
        sanitize_filename,
        GENERATED_AUDIO_NAMES,
        GENERATED_VIDEO_NAMES,
    )
    from core._2_import_subtitles import add_input_subtitles

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        if existing == "archive":
            archive_output()
        elif existing == "replace":
            shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(exist_ok=True)
    if urlparse(source).scheme in {"http", "https"}:
        download_video_ytdlp(source, resolution=load_key("ytb_resolution"))
    elif Path(source).suffix.lower() == ".srt":
        subtitles = Path(source)
    else:
        source = Path(source)
        name = sanitize_filename(source.stem) + source.suffix.lower()
        if name.lower() in GENERATED_AUDIO_NAMES | GENERATED_VIDEO_NAMES:
            name = "input_" + name
        destination = OUTPUT / name
        shutil.copy2(source, destination)
        kind = (
            "video"
            if source.suffix.lower()[1:] in load_key("allowed_video_formats")
            else "audio"
        )
        write_input_manifest(str(destination), kind)
    if subtitles:
        add_input_subtitles(subtitles.name, subtitles.read_bytes())


@app.post("/input", status_code=202)
def set_input(request: InputRequest):
    """Prepare input in the background; poll /status before POST /run."""
    with operation_lock:
        require_idle()
        parsed = urlparse(request.source)
        if parsed.scheme in {"http", "https"}:
            if not parsed.netloc:
                raise HTTPException(422, "Invalid URL")
            source = request.source
        else:
            source = Path(request.source).resolve()
            if not source.is_file():
                raise HTTPException(422, "Source file does not exist")
            if source.is_relative_to(OUTPUT.resolve()):
                raise HTTPException(
                    422,
                    "Source must be outside output/; use /run for an existing input.",
                )
            formats = load_key("allowed_video_formats") + load_key(
                "allowed_audio_formats"
            )
            if source.suffix.lower() == ".srt":
                source = subtitle_file(source)
            elif source.suffix.lower()[1:] not in formats:
                raise HTTPException(422, "Unsupported media format")
            source = str(source)
        subtitles = subtitle_file(request.subtitles) if request.subtitles else None
        if OUTPUT.exists() and any(OUTPUT.iterdir()) and request.existing == "reject":
            raise HTTPException(
                409, "output/ is not empty; explicitly choose archive or replace."
            )
        runner.start(
            [
                (
                    "Prepare input",
                    lambda: prepare_input(source, request.existing, subtitles),
                )
            ]
        )
        return {"accepted": True}


@app.post("/run", status_code=202)
def run(request: RunRequest):
    """Run or retry the shared pipeline. Parameters are saved to config.yaml."""
    with operation_lock:
        require_idle()
        from core._1_ytdlp import find_media_file

        try:
            _, kind = find_media_file()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        needs_dubbing = request.stage == "dubbing" or (
            request.stage == "all" and request.dubbing
        )
        if needs_dubbing and kind in ("audio", "subtitle"):
            raise HTTPException(
                422, "Input without a video supports subtitles only, as in the UI."
            )
        if request.stage == "dubbing" and not all(
            (OUTPUT / name).exists() for name in ("src.srt", "trans.srt")
        ):
            raise HTTPException(409, "Generate subtitles before starting dubbing.")
        if request.target_language is not None:
            update_key("target_language", request.target_language)
        if request.source_language is not None:
            update_key("whisper.language", request.source_language)
        runner.start(get_steps(request.stage, request.dubbing))
        return {"accepted": True}


@app.get("/status")
def status():
    """In-memory execution state; output files survive server restarts."""
    files = sorted(
        p.name for p in OUTPUT.glob("*") if p.is_file() and not p.name.startswith(".")
    )
    return {
        "state": runner.state,
        "active": runner.is_active,
        "step": runner.current_label or None,
        "step_index": runner.current_step,
        "total_steps": runner.total_steps,
        "progress": runner.progress,
        "error": runner.error_msg or None,
        "pause_message": runner.pause_message or None,
        "files": files,
    }


@app.post("/resume")
def resume():
    """Continue a paused task, such as the checkpoints of pause_before_translate and pause_after_translate."""
    with operation_lock:
        if runner.state != "paused":
            raise HTTPException(409, "No task is paused.")
        runner.resume()
        return {"state": runner.state}


@app.post("/stop")
def stop():
    """Cooperative stop: wait until active=false before another operation."""
    with operation_lock:
        runner.stop()
        return {"state": runner.state}


@app.get("/files/{name}")
def file(name: str):
    path = (OUTPUT / name).resolve()
    if path.parent != OUTPUT.resolve() or name.startswith(".") or not path.is_file():
        raise HTTPException(404, "Output file not found")
    return FileResponse(path, filename=path.name)


@app.post("/archive", status_code=202)
def archive():
    """Move output into history/ using the existing archive behavior."""
    with operation_lock:
        require_idle()
        if not OUTPUT.exists() or not any(OUTPUT.iterdir()):
            raise HTTPException(409, "No output to archive")
        runner.start([("Archive output", archive_output)])
        return {"accepted": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("VIDEOLINGO_API_PORT", "8501"))
    uvicorn.run(app, host="127.0.0.1", port=port)

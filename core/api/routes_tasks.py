from __future__ import annotations

import io
from pathlib import Path
import shutil
from typing import Any, Literal
from urllib.parse import quote, urlparse
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from core.api.helpers import (
    _ORPHAN_ACTIVE_STATES,
    _clean_user_path,
    _resolve_workspace,
    _workspace_summary,
    _wrap_workspace_steps,
    get_output_dir,
    get_pipeline_steps,
    get_runner,
    reveal_in_file_manager,
    subtitle_file,
)
from core.api.schemas import (
    CreateTaskRequest,
    InvalidateStageRequest,
    RunRequest,
    TaskLogEntryRequest,
)
from core.utils.config_utils import load_key, load_key_or, update_key
from core.workspace import (
    DEFAULT_GLOSSARY_DOMAIN,
    create_task_workspace,
    list_task_workspaces,
    rmtree_robust,
)

router = APIRouter()


@router.get("/api/tasks")
def list_tasks():
    workspaces = list_task_workspaces(get_output_dir())
    return {"tasks": [_workspace_summary(ws) for ws in workspaces]}


@router.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    ws = _resolve_workspace(task_id)
    return _workspace_summary(ws)


@router.post("/api/tasks", status_code=202)
def create_task(request: CreateTaskRequest):
    out_dir = get_output_dir()
    runner = get_runner()
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
        ws = create_task_workspace(request.title or "youtube_video", base_dir=out_dir)
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
            dl_steps.extend(get_pipeline_steps(request.stage, needs_dub))
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

    ws = create_task_workspace(request.title or src_path.stem, base_dir=out_dir)
    manifest = ws.init_from_local_file(
        src_path,
        subtitle_path=sub_path,
        title=request.title or src_path.stem,
        config_override=overrides,
    )
    needs_dub = request.stage == "dubbing" or (
        request.stage == "all" and request.dubbing
    )
    manifest.setdefault("execution", {}).update(
        {
            "desired_stage": request.stage,
            "dubbing_enabled": needs_dub,
        }
    )
    ws.save_manifest(manifest)
    if request.autorun:
        steps = _wrap_workspace_steps(
            ws, get_pipeline_steps(request.stage, needs_dub), overrides
        )
        runner.enqueue(steps, task_id=ws.task_id)
    return _workspace_summary(ws)


@router.post("/api/tasks/upload", status_code=202)
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
    out_dir = get_output_dir()
    runner = get_runner()
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

    ws = create_task_workspace(stem, base_dir=out_dir)
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
    manifest = ws.init_from_local_file(
        dst_path,
        subtitle_path=sub_local_path,
        title=stem,
        config_override=overrides,
    )
    if sub_bytes is not None:
        (ws.root / "source.srt").write_bytes(sub_bytes)
        manifest["subtitles"] = {
            "source_type": "uploaded_srt",
            "original_path": subtitles.filename if subtitles else "source.srt",
            "filename": "source.srt",
        }
        manifest.setdefault("media", {})["input_subtitle"] = "source.srt"
    needs_dub = stage in ("all", "dubbing")
    manifest.setdefault("execution", {}).update(
        {
            "desired_stage": stage,
            "dubbing_enabled": needs_dub,
        }
    )
    ws.save_manifest(manifest)
    if autorun:
        steps = _wrap_workspace_steps(
            ws, get_pipeline_steps(stage, needs_dub), overrides
        )
        runner.enqueue(steps, task_id=ws.task_id)
    return _workspace_summary(ws)


@router.post("/api/tasks/{task_id}/run", status_code=202)
def run_task(task_id: str, request: RunRequest):
    runner = get_runner()
    ws = _resolve_workspace(task_id)
    if (runner.active_task_id == ws.task_id and runner.is_active) or runner.is_queued(
        ws.task_id
    ):
        raise HTTPException(409, f"Task {ws.task_id} is already running or queued.")
    manifest = ws.load_manifest()
    exec_info = manifest.get("execution") or {}
    saved_stage = exec_info.get("desired_stage")
    effective_stage: Literal["transcribe", "subtitles", "burn", "dubbing", "all"] = (
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
    if (needs_dubbing or effective_stage == "burn") and media_kind in (
        "audio",
        "subtitle",
    ):
        raise HTTPException(
            422, "Input without a video supports subtitles only, as in the UI."
        )
    if effective_stage == "burn" and not all(
        (ws.root / name).exists() for name in ("src.srt", "trans.srt")
    ):
        raise HTTPException(409, "Generate subtitles before burning video.")
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
    if effective_stage != "burn":
        manifest.setdefault("execution", {}).update(
            {
                "desired_stage": effective_stage,
                "dubbing_enabled": needs_dubbing,
            }
        )
    ws.save_manifest(manifest)

    raw_steps = list(get_pipeline_steps(effective_stage, needs_dubbing))
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


@router.post("/api/tasks/{task_id}/pause")
def pause_task(task_id: str):
    runner = get_runner()
    ws = _resolve_workspace(task_id)
    if runner.active_task_id != ws.task_id or not runner.is_active:
        raise HTTPException(409, f"Task {ws.task_id} is not the active running task.")
    runner.pause("Paused by user")
    return {"state": runner.state}


@router.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str):
    runner = get_runner()
    ws = _resolve_workspace(task_id)
    if runner.active_task_id != ws.task_id or runner.state != "paused":
        raise HTTPException(409, f"Task {ws.task_id} is not paused.")
    runner.resume()
    return {"state": runner.state}


@router.post("/api/tasks/{task_id}/stop")
def stop_task(task_id: str):
    runner = get_runner()
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
        runner.stop(clear_queue=False)
        return {"state": runner.state}
    if runner.cancel_queued(ws.task_id):
        ws.update_manifest(execution={"state": "stopped", "pause_reason": None})
        return {"state": "stopped"}
    saved_state = (ws.load_manifest().get("execution") or {}).get("state")
    if saved_state in _ORPHAN_ACTIVE_STATES:
        ws.update_manifest(execution={"state": "stopped", "pause_reason": None})
        return {"state": "stopped"}
    raise HTTPException(409, f"Task {ws.task_id} is not active.")


@router.post("/api/tasks/{task_id}/invalidate")
def invalidate_task_stage(task_id: str, request: InvalidateStageRequest):
    runner = get_runner()
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
            ws, get_pipeline_steps(request.run_stage, needs_dub), overrides
        )
        runner.enqueue(steps, task_id=ws.task_id)
    return {"invalidated": invalidated}


@router.get("/api/tasks/{task_id}/logs")
def get_task_logs(task_id: str):
    ws = _resolve_workspace(task_id)
    return {"task_id": ws.task_id, "logs": ws.load_logs()}


@router.post("/api/tasks/{task_id}/logs", status_code=201)
def append_task_log(task_id: str, request: TaskLogEntryRequest):
    ws = _resolve_workspace(task_id)
    entry = ws.append_log(
        request.msg,
        tag=request.tag,
        label=request.label.upper()[:8],
        warn=request.warn,
        time_str=request.time,
    )
    return {"task_id": ws.task_id, "entry": entry}


@router.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    runner = get_runner()
    ws = _resolve_workspace(task_id)
    if runner.active_task_id == ws.task_id and runner.is_active:
        raise HTTPException(409, "Cannot delete an active task; stop it first.")
    runner.cancel_queued(ws.task_id)
    runner.clear_logs(ws.task_id)
    ws.clear_runtime_cache()
    rmtree_robust(ws.root)
    return {"deleted": ws.task_id}


@router.get("/api/tasks/{task_id}/files/{name:path}")
def get_task_file(task_id: str, name: str, inline: bool = False):
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
    if inline:
        return FileResponse(path, content_disposition_type="inline")
    title = ws.load_manifest().get("title") or ws.task_id
    safe_title = (
        "".join(c for c in str(title) if c not in '<>:"/\\|?*').strip() or "video"
    )
    dl_name = (
        f"{safe_title}_{path.name}" if path.name.startswith("output_") else path.name
    )
    return FileResponse(path, filename=dl_name)


@router.get("/api/tasks/{task_id}/subtitles.zip")
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


@router.post("/api/tasks/{task_id}/open-dir")
def open_task_dir(task_id: str):
    ws = _resolve_workspace(task_id)
    abs_path = ws.root.resolve()
    reveal_in_file_manager(abs_path)
    return {"task_id": ws.task_id, "path": str(abs_path)}

from __future__ import annotations

import contextlib
from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.api.helpers import _clean_user_path, get_output_dir, reveal_in_file_manager
from core.api.schemas import StandaloneBurnRequest
from core.utils.config_utils import load_key

router = APIRouter()

_BURN_SUBTITLE_EXTS = {".srt", ".ass", ".ssa"}


def _get_burn_job():
    import api as _api

    return _api.burn_job


def _validate_burn_subtitle_bytes(payload: bytes, ext: str) -> None:
    if ext == ".srt":
        from core._2_import_subtitles import read_cues

        read_cues(payload)
        return
    try:
        text = payload.decode(
            "utf-16" if payload.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            "字幕文件并非 UTF-8 / UTF-16 编码，请转存为 UTF-8 后重试。"
        ) from exc
    lowered = text.lower()
    if "[events]" not in lowered or "dialogue:" not in lowered:
        raise ValueError("ASS/SSA 字幕文件缺少有效的 [Events] Dialogue 字幕行。")


def _resolve_burn_subtitle_path(raw: str) -> Path:
    path = Path(_clean_user_path(raw)).resolve()
    ext = path.suffix.lower()
    if not path.is_file() or ext not in _BURN_SUBTITLE_EXTS:
        raise HTTPException(
            422, f"字幕文件不存在或格式不支持（需 .srt / .ass / .ssa）: {path}"
        )
    try:
        _validate_burn_subtitle_bytes(path.read_bytes(), ext)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return path


def _resolve_burn_output_path(
    raw_output: str | None, default_dir: Path, default_stem: str, video_path: Path
) -> Path:
    if raw_output and _clean_user_path(raw_output):
        out_path = Path(_clean_user_path(raw_output)).expanduser()
        if not out_path.suffix:
            out_path = out_path.with_suffix(".mp4")
        elif out_path.suffix.lower() != ".mp4":
            raise HTTPException(422, "烧录输出文件目前仅支持 .mp4 格式。")
        try:
            out_path = out_path.resolve()
        except OSError as exc:
            raise HTTPException(422, f"输出路径无效: {exc}") from exc
    else:
        out_path = (default_dir / f"{default_stem}_sub.mp4").resolve()
    if out_path.exists() and out_path.is_dir():
        raise HTTPException(422, f"输出路径不能是已存在的目录: {out_path}")
    for anc in out_path.parents:
        if anc.exists():
            if not anc.is_dir():
                raise HTTPException(422, f"输出路径的父级不是有效目录: {anc}")
            break
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(422, f"无法创建输出目录 {out_path.parent}: {exc}") from exc
    if out_path == video_path.resolve():
        raise HTTPException(422, "输出路径不能与源视频路径相同，避免覆盖原文件。")
    return out_path


@router.get("/api/burn/status")
def get_burn_status():
    return _get_burn_job().snapshot()


@router.post("/api/burn/start", status_code=202)
def start_standalone_burn(request: StandaloneBurnRequest):
    vid_path = Path(_clean_user_path(request.video_path)).resolve()
    if not vid_path.is_file():
        raise HTTPException(422, f"源视频文件不存在: {vid_path}")
    video_formats = {
        str(x).lower().lstrip(".") for x in load_key("allowed_video_formats")
    }
    if vid_path.suffix.lower().lstrip(".") not in video_formats:
        raise HTTPException(422, f"不支持的视频格式: {vid_path.suffix}")

    sub_path = _resolve_burn_subtitle_path(request.subtitle_path)
    sec_sub_path = (
        _resolve_burn_subtitle_path(request.secondary_subtitle_path)
        if request.secondary_subtitle_path and request.secondary_subtitle_path.strip()
        else None
    )
    out_path = _resolve_burn_output_path(
        request.output_path, vid_path.parent, vid_path.stem, vid_path
    )
    return _get_burn_job().start(
        vid_path, sub_path, sec_sub_path, out_path, request.ffmpeg_gpu
    )


@router.post("/api/burn/upload", status_code=202)
def upload_standalone_burn(
    video: UploadFile | None = File(None),
    video_path: str | None = Form(None),
    subtitle: UploadFile | None = File(None),
    subtitle_path: str | None = Form(None),
    secondary_subtitle: UploadFile | None = File(None),
    secondary_subtitle_path: str | None = Form(None),
    output_path: str | None = Form(None),
    ffmpeg_gpu: bool | None = Form(None),
):
    job = _get_burn_job()
    if job.is_running:
        raise HTTPException(409, "当前已有烧录任务正在运行，请等待完成或先停止。")

    out_root = get_output_dir()
    video_formats = {
        str(x).lower().lstrip(".") for x in load_key("allowed_video_formats")
    }
    staging_root = (out_root / "_standalone_burn").resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / f"run_{uuid.uuid4().hex[:12]}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        if video is not None and video.filename:
            raw_vname = video.filename
            v_ext = Path(raw_vname).suffix.lower() or ".mp4"
            v_stem = Path(raw_vname).stem or "video"
            if v_ext.lstrip(".") not in video_formats:
                raise HTTPException(422, f"不支持的视频格式: {v_ext}")
            resolved_vid = staging_dir / f"source{v_ext}"
            with open(resolved_vid, "wb") as out_f:
                shutil.copyfileobj(video.file, out_f)
            default_out_dir = out_root.resolve()
            out_root.mkdir(parents=True, exist_ok=True)
        elif video_path and _clean_user_path(video_path):
            resolved_vid = Path(_clean_user_path(video_path)).resolve()
            if not resolved_vid.is_file():
                raise HTTPException(422, f"源视频文件不存在: {resolved_vid}")
            if resolved_vid.suffix.lower().lstrip(".") not in video_formats:
                raise HTTPException(422, f"不支持的视频格式: {resolved_vid.suffix}")
            v_stem = resolved_vid.stem
            default_out_dir = resolved_vid.parent
        else:
            raise HTTPException(422, "请提供待烧录的视频文件。")

        if subtitle is not None and subtitle.filename:
            s_ext = Path(subtitle.filename).suffix.lower() or ".srt"
            if s_ext not in _BURN_SUBTITLE_EXTS:
                raise HTTPException(422, f"不支持的字幕格式: {s_ext}")
            sub_bytes = subtitle.file.read()
            try:
                _validate_burn_subtitle_bytes(sub_bytes, s_ext)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            resolved_sub = staging_dir / f"primary{s_ext}"
            resolved_sub.write_bytes(sub_bytes)
        elif subtitle_path and _clean_user_path(subtitle_path):
            resolved_sub = _resolve_burn_subtitle_path(subtitle_path)
        else:
            raise HTTPException(422, "请提供主字幕文件 (.srt / .ass / .ssa)。")

        resolved_sec: Path | None = None
        if secondary_subtitle is not None and secondary_subtitle.filename:
            sec_ext = Path(secondary_subtitle.filename).suffix.lower() or ".srt"
            if sec_ext not in _BURN_SUBTITLE_EXTS:
                raise HTTPException(422, f"不支持的副字幕格式: {sec_ext}")
            sec_bytes = secondary_subtitle.file.read()
            try:
                _validate_burn_subtitle_bytes(sec_bytes, sec_ext)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            resolved_sec = staging_dir / f"secondary{sec_ext}"
            resolved_sec.write_bytes(sec_bytes)
        elif secondary_subtitle_path and _clean_user_path(secondary_subtitle_path):
            resolved_sec = _resolve_burn_subtitle_path(secondary_subtitle_path)

        out_path = _resolve_burn_output_path(
            output_path, default_out_dir, v_stem, resolved_vid
        )
        snap = job.start(resolved_vid, resolved_sub, resolved_sec, out_path, ffmpeg_gpu)
    except BaseException:
        with contextlib.suppress(OSError):
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    for old_run in staging_root.glob("run_*"):
        if old_run != staging_dir and old_run.is_dir():
            if not any(
                p and old_run in Path(p).resolve().parents
                for p in (str(resolved_vid), str(resolved_sub), str(resolved_sec or ""))
                if p
            ):
                with contextlib.suppress(OSError):
                    shutil.rmtree(old_run, ignore_errors=True)
    return snap


@router.post("/api/burn/stop")
def stop_standalone_burn():
    return _get_burn_job().stop()


@router.post("/api/burn/open-output")
def open_burn_output():
    snap = _get_burn_job().snapshot()
    target_str = snap.get("completed_output_path") or snap.get("output_path") or ""
    if not target_str:
        raise HTTPException(404, "尚未生成输出文件路径。")
    target = Path(target_str).resolve()
    reveal_file = target.is_file()
    folder = target.parent
    if not reveal_file and not folder.is_dir():
        raise HTTPException(404, f"输出目录不存在: {folder}")
    reveal_in_file_manager(target if reveal_file else folder, select_file=reveal_file)
    return {"path": str(target if reveal_file else folder)}


@router.get("/api/burn/preview")
def preview_burn_output():
    snap = _get_burn_job().snapshot()
    completed = snap.get("completed_output_path") or ""
    if not completed:
        raise HTTPException(404, "当前尚未生成可预览的烧录成片。")
    out_file = Path(completed).resolve()
    if not out_file.is_file():
        raise HTTPException(404, f"成片文件不存在: {out_file}")
    return FileResponse(
        out_file,
        media_type="video/mp4",
        filename=out_file.name,
        content_disposition_type="inline",
    )

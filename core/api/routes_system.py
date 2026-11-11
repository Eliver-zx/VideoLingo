from __future__ import annotations

import copy
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from core.api.helpers import get_output_dir, reveal_in_file_manager
from core.api.schemas import (
    ConfigUpdateRequest,
    DetectMaiRegionRequest,
    LLMProbeRequest,
    PickFilesRequest,
)
import core.utils.config_utils as _cfg
from core.utils.config_utils import (
    ensure_config_schema,
    load_key,
    task_config_context,
    update_key,
    update_keys,
)

router = APIRouter()

WEBUI_DIR = Path(__file__).resolve().parent.parent / "webui"
WEBUI_HTML_PATH = WEBUI_DIR / "index.html"

_SAFE_ASSET_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@router.get("/", response_class=HTMLResponse)
def webui_index():
    if not WEBUI_HTML_PATH.is_file():
        raise HTTPException(404, "Web UI HTML not found")
    return HTMLResponse(WEBUI_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/favicon.ico", include_in_schema=False)
def webui_favicon():
    return Response(status_code=204)


@router.post("/api/dialogs/pick-files")
def pick_local_files(request: PickFilesRequest | None = Body(default=None)):
    """Open native OS file or folder picker for local media, subtitle, cookies, or directory selection."""
    kind = request.kind if request is not None else "media"
    multi = request.multi if request is not None else True
    raw_default_name = (
        re.sub(r'[\\/:*?"<>|\r\n\'`]', "_", (request.default_name or "").strip())
        if request is not None
        else ""
    )
    default_mp4_name = raw_default_name or "output_sub.mp4"
    if not default_mp4_name.lower().endswith(".mp4"):
        default_mp4_name = f"{default_mp4_name}.mp4"
    sys_name = platform.system()
    if sys_name == "Darwin":
        if kind == "directory":
            script = (
                'set chosenFolder to choose folder with prompt "选择目录 - VideoLingo"\n'
                "return POSIX path of chosenFolder"
            )
        elif kind == "save_video":
            script = (
                'set chosenFile to choose file name with prompt "选择输出 MP4 保存位置 - VideoLingo" '
                f'default name "{default_mp4_name}"\n'
                "return POSIX path of chosenFile"
            )
        else:
            prompts = {
                "video": "选择待烧录的视频文件 - VideoLingo",
                "subtitle": "选择字幕文件 (.srt / .ass / .ssa) - VideoLingo",
                "cookies": "选择 Cookies 文件 (.txt) - VideoLingo",
                "media": "选择媒体或字幕文件 - VideoLingo",
            }
            prompt_text = prompts[kind]
            if multi:
                script = (
                    f'set chosenFiles to choose file with prompt "{prompt_text}" '
                    "with multiple selections allowed\n"
                    "set posixPaths to {}\n"
                    "repeat with aFile in chosenFiles\n"
                    "  set end of posixPaths to POSIX path of aFile\n"
                    "end repeat\n"
                    'set AppleScript\'s text item delimiters to "\\n"\n'
                    "return posixPaths as text"
                )
            else:
                script = (
                    f'set chosenFile to choose file with prompt "{prompt_text}"\n'
                    "return POSIX path of chosenFile"
                )
        proc = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            return {"paths": []}
        paths = [p.strip() for p in (proc.stdout or "").splitlines() if p.strip()]
        if kind == "directory":
            paths = [p.rstrip("/") if p != "/" else p for p in paths]
        elif kind == "save_video":
            paths = [p if p.lower().endswith(".mp4") else f"{p}.mp4" for p in paths]
        return {"paths": paths}
    if sys_name == "Windows":
        top_form = (
            "$top = New-Object System.Windows.Forms.Form; "
            "$top.FormBorderStyle = 'None'; "
            "$top.ShowInTaskbar = $false; "
            "$top.Opacity = 0; "
            "$top.StartPosition = 'Manual'; "
            "$top.Location = New-Object System.Drawing.Point(-32000, -32000); "
            "$top.Width = 1; $top.Height = 1; "
            "$top.TopMost = $true; "
            "$top.Show(); $top.Activate(); "
        )
        if kind == "directory":
            ps_script = (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                "$f.Description = '选择目录 - VideoLingo'; "
                f"{top_form}"
                "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.SelectedPath }; "
                "$top.Dispose()"
            )
        elif kind == "save_video":
            ps_script = (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$f = New-Object System.Windows.Forms.SaveFileDialog; "
                "$f.Title = '选择输出 MP4 保存位置 - VideoLingo'; "
                "$f.Filter = 'MP4 视频 (*.mp4)|*.mp4|所有文件 (*.*)|*.*'; "
                "$f.DefaultExt = 'mp4'; "
                f"$f.FileName = '{default_mp4_name}'; "
                f"{top_form}"
                "if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }; "
                "$top.Dispose()"
            )
        else:
            win_filters = {
                "video": "视频文件 (*.mp4;*.mov;*.mkv;*.avi;*.flv;*.wmv;*.webm)|*.mp4;*.mov;*.mkv;*.avi;*.flv;*.wmv;*.webm|所有文件 (*.*)|*.*",
                "subtitle": "字幕文件 (*.srt;*.ass;*.ssa)|*.srt;*.ass;*.ssa|所有文件 (*.*)|*.*",
                "cookies": "Cookies 文件 (*.txt)|*.txt|所有文件 (*.*)|*.*",
                "media": "媒体与字幕文件 (*.mp4;*.mov;*.mkv;*.avi;*.flv;*.wmv;*.webm;*.wav;*.mp3;*.flac;*.m4a;*.srt)|*.mp4;*.mov;*.mkv;*.avi;*.flv;*.wmv;*.webm;*.wav;*.mp3;*.flac;*.m4a;*.srt|所有文件 (*.*)|*.*",
            }
            multiselect = "$true" if multi else "$false"
            out_prop = "FileNames" if multi else "FileName"
            ps_script = (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$f = New-Object System.Windows.Forms.OpenFileDialog; "
                "$f.Title = '选择文件 - VideoLingo'; "
                f"$f.Filter = '{win_filters[kind]}'; "
                f"$f.Multiselect = {multiselect}; "
                f"{top_form}"
                f"if ($f.ShowDialog($top) -eq [System.Windows.Forms.DialogResult]::OK) {{ $f.{out_prop} | ForEach-Object {{ Write-Output $_ }} }}; "
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


@router.post("/api/mai/detect-region")
def detect_mai_region(request: DetectMaiRegionRequest):
    from core.asr_backend.mai_asr import detect_region

    region = detect_region(request.api_key.strip())
    if region:
        update_key("whisper.mai_region", region, add_missing=True)
    return {"region": region}


@router.post("/api/llm/models")
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


@router.post("/api/llm/check")
def check_llm_api(request: LLMProbeRequest):
    import api as _api
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
                f"This is a connectivity test ({int(_api.time.time())}), respond with {{'message':'success'}} in JSON format.",
                resp_type="json",
                log_title="None",
            )
        if isinstance(resp, dict) and resp.get("message") == "success":
            return {"valid": True, "error": ""}
        return {"valid": False, "error": f"Unexpected response: {str(resp)[:240]}"}
    except Exception as exc:
        return {"valid": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}


@router.get("/api/config")
def get_config():
    ensure_config_schema(_cfg.CONFIG_PATH)
    return copy.deepcopy(_cfg._load_cached_config_dict(_cfg.CONFIG_PATH))


@router.put("/api/config")
def put_config(request: ConfigUpdateRequest):
    update_keys(request.updates, add_missing=True)
    return {"updated": list(request.updates.keys())}


@router.post("/api/output/open-dir")
def open_output_root_dir():
    out_dir = get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_path = out_dir.resolve()
    reveal_in_file_manager(abs_path)
    return {"path": str(abs_path)}


def _serve_webui_subdir_file(subdir: str, filename: str, suffix: str, media_type: str):
    if (
        not _SAFE_ASSET_FILENAME_RE.fullmatch(filename)
        or filename.startswith(".")
        or not filename.endswith(suffix)
    ):
        raise HTTPException(404, "Not found")
    base_dir = (WEBUI_DIR / subdir).resolve()
    asset_path = (base_dir / filename).resolve()
    if asset_path.parent != base_dir or not asset_path.is_file():
        raise HTTPException(404, f"Web UI asset not found: {subdir}/{filename}")
    return FileResponse(
        asset_path,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/js/{filename}")
def webui_js_asset(filename: str):
    return _serve_webui_subdir_file(
        "js", filename, ".js", "application/javascript; charset=utf-8"
    )


@router.get("/css/{filename}")
def webui_css_asset(filename: str):
    return _serve_webui_subdir_file("css", filename, ".css", "text/css; charset=utf-8")

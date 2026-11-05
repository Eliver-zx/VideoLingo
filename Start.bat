@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONWARNINGS=ignore"
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%USERPROFILE%\.cargo\bin" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
if exist "%USERPROFILE%\scoop\shims" set "PATH=%USERPROFILE%\scoop\shims;%PATH%"
if exist "C:\ProgramData\chocolatey\bin" set "PATH=C:\ProgramData\chocolatey\bin;%PATH%"
cd /D "%~dp0"

if "%VIDEOLINGO_API_PORT%"=="" set "VIDEOLINGO_API_PORT=8501"
set "URL=http://127.0.0.1:%VIDEOLINGO_API_PORT%"

where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is not installed or not in PATH. Install it from https://docs.astral.sh/uv/
    goto failed
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo ERROR: ffmpeg is not installed or not in PATH. Install via: winget install "FFmpeg ^(Essentials Build^)"
    goto failed
)

where ffprobe >nul 2>nul
if errorlevel 1 (
    echo ERROR: ffprobe is not installed or not in PATH. Install via: winget install "FFmpeg ^(Essentials Build^)"
    goto failed
)

if exist "%USERPROFILE%\.venvs\videolingo\Scripts\python.exe" (
    set "VENV_DIR=%USERPROFILE%\.venvs\videolingo"
) else (
    set "VENV_DIR=%CD%\.venv"
)
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [1/2] Creating Python 3.12 virtual environment at "%VENV_DIR%" ...
    uv venv "%VENV_DIR%" --python 3.12
    if errorlevel 1 goto failed
)

"%VENV_PY%" -c "import fastapi, multipart, uvicorn, openai, ruamel.yaml, demucs, rapidfuzz, qwen_asr, torch, importlib.metadata as m; assert torch.cuda.is_available(); assert tuple(map(int, m.version('nagisa').split('.')[:2])) >= (0, 3)" >nul 2>nul
if errorlevel 1 (
    echo [2/2] Installing dependencies ^(CUDA PyTorch + Qwen3-ASR^) via uv ...
    uv pip install --python "%VENV_PY%" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match -r requirements.txt
    if errorlevel 1 goto failed
)

"%VENV_PY%" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available in PyTorch! Check NVIDIA driver and CUDA PyTorch install.'; print(f'GPU ready: {torch.cuda.get_device_name(0)}')"
if errorlevel 1 goto failed

echo Using Python: "%VENV_PY%"
echo Starting VideoLingo WebUI at %URL% ...
start "" powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 50; $i++) { try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', %VIDEOLINGO_API_PORT%); $c.Close(); Start-Process '%URL%'; exit } catch { Start-Sleep -Milliseconds 200 } }"
"%VENV_PY%" api.py
goto end

:failed
echo Installation or launch failed.
pause
exit /b 1

:end
pause

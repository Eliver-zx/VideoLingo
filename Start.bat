@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONWARNINGS=ignore"
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%USERPROFILE%\.cargo\bin" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cd /D "%~dp0"

if "%VIDEOLINGO_API_PORT%"=="" set "VIDEOLINGO_API_PORT=8501"

where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is not installed or not in PATH. Install it from https://docs.astral.sh/uv/
    goto failed
)

if exist "%USERPROFILE%\.venvs\videolingo\Scripts\python.exe" (
    set "VENV_DIR=%USERPROFILE%\.venvs\videolingo"
) else (
    set "VENV_DIR=%CD%\.venv"
)
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYW=%VENV_DIR%\Scripts\pythonw.exe"

if not exist "%VENV_PY%" (
    echo [1/2] Creating Python 3.12 virtual environment at "%VENV_DIR%" ...
    uv venv "%VENV_DIR%" --python 3.12
    if errorlevel 1 goto failed
)

"%VENV_PY%" -c "import fastapi, uvicorn, openai, ruamel.yaml, spacy, demucs, qwen_asr, torch, pystray, PIL" >nul 2>nul
if errorlevel 1 (
    echo [2/2] Installing dependencies ^(CUDA PyTorch + Qwen3-ASR + System Tray^) via uv ...
    uv pip install --python "%VENV_PY%" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match -r requirements.txt
    if errorlevel 1 goto failed
)

if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" -m core.tray
    exit /b 0
)

start "" /B "%VENV_PY%" -m core.tray
exit /b 0

:failed
echo Installation or launch failed.
pause
exit /b 1

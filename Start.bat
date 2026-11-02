@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONWARNINGS=ignore"
cd /D "%~dp0"

if "%VIDEOLINGO_API_PORT%"=="" set "VIDEOLINGO_API_PORT=8501"
set "URL=http://127.0.0.1:%VIDEOLINGO_API_PORT%"

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

"%VENV_PY%" -c "import fastapi, uvicorn, openai, ruamel.yaml, spacy, demucs, qwen_asr, torch" >nul 2>nul
if errorlevel 1 (
    echo [2/2] Installing dependencies ^(CUDA PyTorch + Qwen3-ASR^) via uv ...
    uv pip install --python "%VENV_PY%" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match -r requirements.txt
    if errorlevel 1 goto failed
)

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

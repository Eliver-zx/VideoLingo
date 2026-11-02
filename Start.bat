@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONWARNINGS=ignore"
cd /D "%~dp0"

if "%VIDEOLINGO_API_PORT%"=="" set "VIDEOLINGO_API_PORT=8501"
set "URL=http://127.0.0.1:%VIDEOLINGO_API_PORT%"
set "VENV_PY="

set "SHARED_VENV=%USERPROFILE%\.venvs\videolingo"
if exist "%SHARED_VENV%\Scripts\python.exe" (
    "%SHARED_VENV%\Scripts\python.exe" -c "import sys; sys.exit(not (sys.version_info[:2] == (3, 12)))" >nul 2>nul
    if not errorlevel 1 (
        set "VENV_PY=%SHARED_VENV%\Scripts\python.exe"
        goto check_deps
    )
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(not (sys.version_info[:2] == (3, 12)))" >nul 2>nul
    if not errorlevel 1 (
        set "VENV_PY=.venv\Scripts\python.exe"
        goto check_deps
    )
)

where conda >nul 2>nul
if not errorlevel 1 (
    call conda activate videolingo >nul 2>nul
    if not errorlevel 1 if /I "!CONDA_DEFAULT_ENV!"=="videolingo" (
        python -c "import sys; sys.exit(not (sys.version_info[:2] == (3, 12)))" >nul 2>nul
        if not errorlevel 1 (
            set "VENV_PY=python"
            goto check_deps
        )
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import fastapi, uvicorn, openai, ruamel.yaml" >nul 2>nul
    if not errorlevel 1 (
        set "VENV_PY=python"
        goto launch
    )
)

echo First run: installing VideoLingo environment...
where uv >nul 2>nul
if errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
uv run --no-project --python 3.12 setup_env.py --yes
if errorlevel 1 goto failed
set "VENV_PY=.venv\Scripts\python.exe"

:check_deps
"%VENV_PY%" -c "import fastapi, uvicorn, openai, ruamel.yaml" >nul 2>nul
if errorlevel 1 (
    echo Installing missing dependencies...
    "%VENV_PY%" installer.py --yes
    if errorlevel 1 goto failed
)

:launch
echo Starting VideoLingo WebUI at %URL% ...
start "" powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 50; $i++) { try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', %VIDEOLINGO_API_PORT%); $c.Close(); Start-Process '%URL%'; exit } catch { Start-Sleep -Milliseconds 200 } }"
"%VENV_PY%" api.py
goto end

:failed
echo Launch or installation failed.
pause
exit /b 1

:end
pause

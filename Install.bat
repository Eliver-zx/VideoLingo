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

where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv is not installed or not in PATH. Install it from https://docs.astral.sh/uv/
    goto failed
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo WARNING: ffmpeg is not in PATH. Install via: winget install "FFmpeg ^(Essentials Build^)"
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
) else (
    echo [1/2] Using existing virtual environment at "%VENV_DIR%" ...
)

echo [2/2] Installing dependencies ^(CUDA PyTorch + Qwen3-ASR^) via uv ...
uv pip install --python "%VENV_PY%" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match -r requirements.txt
if errorlevel 1 goto failed

echo Upgrading nagisa for UTF-8 compatibility on Windows ...
uv pip install --python "%VENV_PY%" --no-deps "nagisa>=0.3.1"
if errorlevel 1 goto failed

"%VENV_PY%" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo.
echo Environment installation complete! Double-click Start.bat to launch VideoLingo.
pause
exit /b 0

:failed
echo.
echo Installation failed.
pause
exit /b 1

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

if exist "%USERPROFILE%\.venvs\videolingo\Scripts\python.exe" (
    set "VENV_PY=%USERPROFILE%\.venvs\videolingo\Scripts\python.exe"
) else if exist "%CD%\.venv\Scripts\python.exe" (
    set "VENV_PY=%CD%\.venv\Scripts\python.exe"
) else (
    echo ERROR: Python virtual environment not found ^(.venv\Scripts\python.exe^).
    echo Please run Install.bat first to set up dependencies.
    pause
    exit /b 1
)

echo Using Python: "%VENV_PY%"
echo Starting VideoLingo WebUI at %URL% ...
start "" powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 50; $i++) { try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', %VIDEOLINGO_API_PORT%); $c.Close(); Start-Process '%URL%'; exit } catch { Start-Sleep -Milliseconds 200 } }"
"%VENV_PY%" api.py
pause

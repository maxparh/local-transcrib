@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Python environment not found: "%~dp0.venv\Scripts\python.exe"
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0desktop.py"
set "desktop_exit_code=%errorlevel%"
if not "%desktop_exit_code%"=="0" (
    echo Desktop exited with code %desktop_exit_code%.
    echo See logs\desktop.log and logs\desktop-crash.log for details.
    pause
)
exit /b %desktop_exit_code%

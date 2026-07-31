@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\batch_trim_gameplay_h5.py --ranges-dir data --out-dir data\cleaned
pause

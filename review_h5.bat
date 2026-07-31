@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Drag a .h5 file onto this bat, or run:
    echo review_h5.bat data\party_kart_corrections_01.h5
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\review_gameplay_h5.py --input "%~1"
pause

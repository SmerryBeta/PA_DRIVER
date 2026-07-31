@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\collect_gameplay.py --out data\party_kart_01.h5 --fps 20 --left 0 --top 170 --width 1920 --height 800 --output-width 224 --output-height 96 --preview
pause

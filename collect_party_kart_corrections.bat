@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "runs\party_kart_bc.pt" (
    echo Missing runs\party_kart_bc.pt
    echo Run train_party_kart.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\collect_gameplay_corrections.py --model runs\party_kart_bc_corrected.pt --out data\party_kart_corrections_01.h5 --fps 20 --left 0 --top 170 --width 1920 --height 800 --preview --preview-topmost
pause

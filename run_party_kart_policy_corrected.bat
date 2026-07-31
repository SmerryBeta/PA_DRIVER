@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "runs\party_kart_bc_corrected.pt" (
    echo Missing runs\party_kart_bc_corrected.pt
    echo Run train_party_kart_with_corrections.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\run_game_policy.py --model runs\party_kart_bc_corrected.pt --fps 20 --left 0 --top 170 --width 1920 --height 800 --preview --preview-topmost --allow-s --no-space --turn-threshold 0.65 --s-threshold 0.45 --stuck-recovery
pause

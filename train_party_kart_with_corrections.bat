@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "data\party_kart_corrections_*.h5" (
    echo Missing data\party_kart_corrections_*.h5
    echo Run collect_party_kart_corrections.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\train_game_bc.py --data data\party_kart*.h5 --out runs\party_kart_bc_corrected.pt --init-model runs\party_kart_bc.pt --epochs 6 --batch-size 128 --lr 0.0001 --stack-size 4 --width 112 --height 48 --labels 0 1 2 3 4 --pos-weight-cap 5
pause

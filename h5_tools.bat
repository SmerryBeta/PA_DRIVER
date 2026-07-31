@echo off
if not "%H5_TOOLS_READY%"=="1" (
    set H5_TOOLS_READY=1
    cmd /k "%~f0"
    exit /b
)

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    pause
    exit /b 1
)

doskey review=".\.venv\Scripts\python.exe" scripts\review_gameplay_h5.py --input $*
doskey trim=".\.venv\Scripts\python.exe" scripts\trim_gameplay_h5.py --input $*
doskey trim_overwrite=".\.venv\Scripts\python.exe" scripts\trim_gameplay_h5.py --input $* --overwrite
doskey batch_trim=".\.venv\Scripts\python.exe" scripts\batch_trim_gameplay_h5.py --ranges-dir data --out-dir data\cleaned $*
doskey combineh5=".\.venv\Scripts\python.exe" scripts\combine_gameplay_h5.py $*
doskey stats=".\.venv\Scripts\python.exe" scripts\h5_stats.py $*
doskey train_corrected=train_party_kart_with_corrections.bat
doskey train_cleaned=train_party_kart_cleaned.bat
doskey train_cleaned_corrected=train_party_kart_cleaned_with_corrections.bat
doskey collect_correct=collect_party_kart_corrections.bat

echo PA_DRIVER H5 tools
echo.
echo Commands:
echo   review data\party_kart_corrections_01.h5
echo   trim data\party_kart_corrections_01.h5
echo   batch_trim
echo   combineh5 data\cleaned\*.h5 --out data\combined_cleaned.h5
echo   stats data\party_kart*.h5
echo   collect_correct
echo   train_corrected
echo   train_cleaned
echo   train_cleaned_corrected
echo.
echo Use Tab to complete file paths. Type exit to close.
echo.

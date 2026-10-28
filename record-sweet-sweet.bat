@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment not found. See docs\human-recording.md.
    pause
    exit /b 1
)
echo Starting the Sweet Sweet human recorder. Keep this window open.
echo If another Dolphin instance is running, close it first.
echo.
".venv\Scripts\python.exe" scripts\record_human_smg1.py
if errorlevel 1 (
    echo.
    echo The recorder stopped with an error. See the message above.
) else (
    echo.
    echo The recorder has stopped. Successful recordings are under runs\human-demonstrations.
)
pause

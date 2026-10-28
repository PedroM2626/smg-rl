@echo off
setlocal
cd /d "%~dp0"
echo Starting SMG2 Spinning and Spinning and Spinning. Keep this window open.
".venv\Scripts\python.exe" scripts\record_human_smg1.py --game smg2 --variation spinning
pause

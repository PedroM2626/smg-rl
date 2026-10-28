@echo off
setlocal
cd /d "%~dp0"
echo Starting SMG2 Turning Turning Double Time. Keep this window open.
".venv\Scripts\python.exe" scripts\record_human_smg1.py --game smg2 --variation double_time
pause

@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment not found. See docs\human-recording.md.
  pause
  exit /b 1
)
echo Center of the Universe recorder: reach the reference location and press F8.
echo F8 saves a goal candidate; an independent replay and visual review are required.
echo Death discards the attempt and reloads slot 3. F9 stops without saving.
echo Close other Dolphin instances before starting.
".venv\Scripts\python.exe" scripts\record_human_smg1.py --game smg1 --variation center_of_universe
if errorlevel 1 echo Recorder stopped with an error. See the message above.
pause

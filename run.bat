@echo off
REM Journaling Mood & Burnout Tracker -- double-click launcher.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.12 -m venv .venv || python -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

if not exist "artifacts\metrics.json" (
  echo Running the study for the first time. This downloads two public text corpora
  echo and fine-tunes a small transformer on CPU -- allow about 30 minutes...
  .venv\Scripts\python.exe build.py --figures
)

echo.
echo Opening http://localhost:8516  --  press Ctrl+C in this window to stop.
echo.
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8516
pause

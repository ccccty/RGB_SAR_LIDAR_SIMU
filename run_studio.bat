@echo off
REM Local dataset studio: click parameters, get a dataset. No network needed.
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
set SIMKIT_CUBE_MB=160
if not exist "env\Scripts\python.exe" (
  echo [!] env\Scripts\python.exe not found - create the venv first:
  echo     python -m venv env ^&^& env\Scripts\pip install numpy
  pause
  exit /b 1
)
echo Starting simkit studio on http://127.0.0.1:8808/
start "" http://127.0.0.1:8808/
"env\Scripts\python.exe" -m simkit studio -p 8808
pause

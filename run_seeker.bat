@echo off
rem Seeker-view dataset: diving + weaving platform, two fields of view, SAR on board.
rem 3 scenes x 2 clips x 120 frames @ 20 fps by default. Optical (Blender) is the
rem slow part; pass --no-optical to generate the numpy sensors only.
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
set SIMKIT_CUBE_MB=192
if not exist "env\Scripts\python.exe" (
  echo [ERROR] env\Scripts\python.exe not found - create the venv first:
  echo     python -m venv env ^&^& env\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
if "%SIMKIT_BLENDER%"=="" (
  for /d %%D in ("third_party\blender*") do set SIMKIT_BLENDER=%%~fD\blender.exe
)
if not exist "%SIMKIT_BLENDER%" set SIMKIT_BLENDER=
if "%SIMKIT_BLENDER%"=="" (
  echo [warn] Blender not found - generating everything except the visible band
  set SENSORS=ir,radar,lidar,sar
  set EXTRA=--no-optical
) else (
  set SENSORS=optical,ir,radar,lidar,sar
  set EXTRA=
)

echo === selftest ===
"env\Scripts\python.exe" -m simkit selftest
if errorlevel 1 exit /b 1

echo.
echo === generate seeker dataset (%SENSORS%) ===
"env\Scripts\python.exe" tools\build_seeker.py -s %SENSORS% %EXTRA% --jobs 2
if errorlevel 1 (
  echo [ERROR] build failed
  pause
  exit /b 1
)

echo.
echo Output in out\seeker_*  -- double-click index.html to browse
pause
endlocal

@echo off
rem One-click demo: self-test + 6-frame all-sensor dataset.
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
if not exist "env\Scripts\python.exe" (
  echo [ERROR] env\Scripts\python.exe not found. Copy the whole folder from the build machine.
  exit /b 1
)
if "%SIMKIT_BLENDER%"=="" (
  for /d %%D in ("third_party\blender-*") do set SIMKIT_BLENDER=%%~fD\blender.exe
)
echo === selftest ===
"env\Scripts\python.exe" -m simkit selftest
if errorlevel 1 exit /b 1
echo.
echo === generate demo (optical needs Blender) ===
if "%SIMKIT_BLENDER%"=="" (
  echo [warn] Blender not found - generating IR + radar only
  "env\Scripts\python.exe" -m simkit gen -c scenes\demo_all.json -o out\demo_all -n 3 -s ir,radar
) else (
  "env\Scripts\python.exe" -m simkit gen -c scenes\demo_all.json -o out\demo_all -n 3 -s optical,ir,radar
)
echo.
echo Output in out\demo_all
endlocal

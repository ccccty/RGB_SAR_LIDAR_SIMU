@echo off
cd /d "%~dp0"
set PYTHONPATH=%~dp0
echo.
echo   ============================================
echo    Dataset Studio  -  http://127.0.0.1:8808
echo    Close this window to stop the server.
echo   ============================================
echo.
start "" http://127.0.0.1:8808
".\env\Scripts\python.exe" -m simkit studio -p 8808
echo.
echo   Server stopped.
pause

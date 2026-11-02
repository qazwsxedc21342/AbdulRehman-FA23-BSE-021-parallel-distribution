@echo off
setlocal
set "SERVER_HOST=0.0.0.0"
set "SERVER_PORT=5000"

title CSC-334: Remote Worker Server Node (Computer B)
cd /d "%~dp0"

echo ======================================================================
echo   CSC-334: REMOTE WORKER SERVER NODE (COMPUTER B)
echo ======================================================================
echo   [1] Launch Modern Server GUI Dashboard (Recommended)
echo   [2] Launch Command-Line Console Daemon (%SERVER_HOST%:%SERVER_PORT%)
echo ======================================================================
set "choice=1"
set /p choice="Select Option [1 or 2] (Default 1): "

if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

if "%choice%"=="2" (
    echo Starting Server CLI Daemon on %SERVER_HOST%:%SERVER_PORT%...
    "%PYTHON_EXE%" -u server_daemon.py --host %SERVER_HOST% --port %SERVER_PORT%
) else (
    echo Starting Modern Server GUI Dashboard...
    "%PYTHON_EXE%" server_gui.py
)

pause
endlocal
exit /b 0

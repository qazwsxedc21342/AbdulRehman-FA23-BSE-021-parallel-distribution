@echo off
setlocal
title CSC-334: Video Offload Client Node (Computer A)
cd /d "%~dp0client"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)
"%PYTHON_EXE%" client_gui.py
pause
endlocal
exit /b 0

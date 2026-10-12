@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo Quantum Teleportation - PyZX + DisCoPy Practical
echo ================================================================

if not exist .venv\Scripts\python.exe (
    echo Creating Python virtual environment...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
)

echo Installing/updating dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Checking environment...
.venv\Scripts\python.exe check_environment.py
if errorlevel 1 goto :error

echo.
echo Running complete practical...
.venv\Scripts\python.exe quantum_teleportation_pyzx_discopy.py
if errorlevel 1 goto :error

echo.
echo SUCCESS. Opening results folder...
start "" "%~dp0results"
echo.
echo To display all figures in windows, run:
echo   .venv\Scripts\python.exe show_results.py
pause
exit /b 0

:error
echo.
echo ERROR: setup or practical execution failed.
echo Read README_VSCODE.md and copy the complete terminal error if you need help.
pause
exit /b 1

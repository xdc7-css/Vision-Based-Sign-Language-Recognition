@echo off
setlocal enabledelayedexpansion

title ArSL Studio — Local Web Application
echo ========================================================
echo  ArSL STUDIO — Arabic Sign Language Recognition (Web)
echo ========================================================
echo.

:: Locate project root
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [!] ERROR: Python virtual environment not found in .venv\
    echo Please install dependencies first.
    pause
    exit /b 1
)

:: Run launcher with web flag directly
".venv\Scripts\python.exe" run.py 2

if %errorlevel% neq 0 (
    echo.
    echo [!] Server exited with error code %errorlevel%
    pause
)


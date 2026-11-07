@echo off
setlocal

cd /d "%~dp0"

:: Check if virtual environment exists
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXEC=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_EXEC=python"
    ) else (
        echo [ERROR] Python environment not found!
        echo Please run SETUP_WINDOWS.bat first to configure your system.
        pause
        exit /b 1
    )
)

:: Forward arguments or launch interactive menu
"%PY_EXEC%" "%~dp0run.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [!] Application exited with error code %errorlevel%.
    pause
)


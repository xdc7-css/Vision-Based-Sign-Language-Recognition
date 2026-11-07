@echo off
setlocal enabledelayedexpansion

title ArSL Studio - Windows Environment Setup
cd /d "%~dp0"

echo =======================================================
echo   ArSL STUDIO - One-Click Environment Setup
echo   Arabic Sign Language Recognition
echo =======================================================
echo.

:: 1. Detect Python
echo [*] Checking Python installation...
set "PY_CMD="

where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PY_VER=%%i"
    if defined PY_VER (
        set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PY_VER=%%i"
        if defined PY_VER (
            set "PY_CMD=py -3.12"
        ) else (
            for /f "tokens=*" %%i in ('py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PY_VER=%%i"
            if defined PY_VER set "PY_CMD=py -3"
        )
    )
)

if not defined PY_CMD (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Detected Python !PY_VER! (!PY_CMD!)
echo.

:: 2. Detect Node.js (Optional for running pre-bundled web, required for development)
echo [*] Checking Node.js installation...
set "NODE_CMD="
where node >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('node -v 2^>nul') do set "NODE_VER=%%i"
    set "NODE_CMD=node"
    echo [OK] Detected Node.js !NODE_VER!
) else (
    echo [NOTE] Node.js was not detected in PATH.
    echo        (Pre-bundled production web assets will still run without Node.js)
)
echo.

:: 3. Setup Virtual Environment
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [*] Creating virtual environment at .venv ...
    !PY_CMD! -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [OK] Existing virtual environment found at .venv
)
echo.

:: 4. Install Python Dependencies
echo [*] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet

echo [*] Installing dependencies from requirements.txt...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo [OK] Python dependencies verified.
echo.

:: 5. Verify / Build Web Frontend
set "DIST_INDEX=%~dp0web\dist\index.html"
if not exist "%DIST_INDEX%" (
    echo [*] Production web bundle not found. Building now...
    if defined NODE_CMD (
        cd /d "%~dp0web"
        echo [*] Installing frontend npm packages...
        call npm ci || call npm install
        echo [*] Building production web assets...
        call npm run build
        cd /d "%~dp0"
        if not exist "%DIST_INDEX%" (
            echo [WARNING] Failed to build web frontend bundle.
        ) else (
            echo [OK] Production web bundle generated successfully.
        )
    ) else (
        echo [WARNING] Node.js is required to build the web frontend.
        echo           Install Node.js 18+ or 20+ from https://nodejs.org/
    )
) else (
    echo [OK] Production web bundle is present and ready.
)
echo.

:: 6. Verify Critical Model Assets & Datasets
echo [*] Verifying essential AI model checkpoints...
"%VENV_PY%" -c "
import os, sys
models = [
    'data/models/karsl/karsl_502.onnx',
    'data/models/karsl/hand_landmarker.task',
    'data/models/karsl/face_landmarker.task',
    'data/models/karsl/pose_landmarker.task',
    'data/models/iraqi/static_model.pkl',
    'data/models/multimodal/static_model.pkl',
]
missing = [m for m in models if not os.path.isfile(m)]
if missing:
    print('[ERROR] Missing essential model assets:', missing, file=sys.stderr)
    sys.exit(1)
print('[OK] All essential AI model checkpoints verified.')
"
if %errorlevel% neq 0 (
    echo [ERROR] Model verification failed.
    pause
    exit /b 1
)

:: 7. Self-Test Quick Health Check
echo [*] Running system sanity check...
"%VENV_PY%" -c "
import src
import backend
from src.infrastructure.dataset.gesture_registry import GestureRegistry
reg = GestureRegistry()
print(f'[OK] System modules and vocabulary registry verified ({len(reg.get_all_classes())} active classes).')
"
if %errorlevel% neq 0 (
    echo [ERROR] Health check failed.
    pause
    exit /b 1
)
echo.

echo =======================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo =======================================================
echo.
echo You can now launch ArSL Studio anytime using:
echo    RUN.bat
echo or:
echo    python run
echo.
pause


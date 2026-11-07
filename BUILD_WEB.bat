@echo off
setlocal enabledelayedexpansion

title ArSL Studio — Build Web Frontend
echo ========================================================
echo  ArSL STUDIO — Building React/TypeScript Frontend
echo ========================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%web"

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ERROR: Node.js / npm is not installed or not in PATH.
    pause
    exit /b 1
)

echo [*] Installing dependencies if needed...
call npm install

echo [*] Compiling TypeScript and bundling Vite assets...
call npm run build

if %errorlevel% equ 0 (
    echo.
    echo [✓] SUCCESS: Production bundle generated in web\dist\
    echo You can now launch the web version using RUN_WEB.bat or 'python run'
) else (
    echo.
    echo [!] Build failed with error code %errorlevel%
)

pause


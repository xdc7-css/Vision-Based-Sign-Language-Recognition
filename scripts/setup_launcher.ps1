# One-time project launcher configuration script for PowerShell
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Write-Host "[*] Configuring Vision-Based Sign Language Recognition Launcher in: $projectRoot" -ForegroundColor Cyan

# Ensure subdirectories have launcher trampolines
$targets = @("src", "src\models", "src\presentation", "tests", "docs")
foreach ($t in $targets) {
    $dir = Join-Path $projectRoot $t
    if (Test-Path $dir) {
        Copy-Item (Join-Path $projectRoot "run.py") (Join-Path $dir "run.py") -Force
        Copy-Item (Join-Path $projectRoot "run") (Join-Path $dir "run") -Force
        Write-Host "  [+] Synced launcher in $t" -ForegroundColor Green
    }
}

# Set User PYTHONPATH via registry
$existing = (Get-ItemProperty -Path 'HKCU:\Environment' -Name 'PYTHONPATH' -ErrorAction SilentlyContinue).PYTHONPATH
if (-not $existing) {
    Set-ItemProperty -Path 'HKCU:\Environment' -Name 'PYTHONPATH' -Value $projectRoot
    Write-Host "  [+] Registered User PYTHONPATH: $projectRoot" -ForegroundColor Green
} elseif ($existing -notmatch [regex]::Escape($projectRoot)) {
    $newVal = "$projectRoot;$existing"
    Set-ItemProperty -Path 'HKCU:\Environment' -Name 'PYTHONPATH' -Value $newVal
    Write-Host "  [+] Updated User PYTHONPATH: $newVal" -ForegroundColor Green
} else {
    Write-Host "  [=] User PYTHONPATH already configured: $existing" -ForegroundColor Green
}

Write-Host "`n[✓] Launcher setup complete! You can now launch from ANY project directory using:" -ForegroundColor Yellow
Write-Host "      python run" -ForegroundColor White
Write-Host "      python run.py`n" -ForegroundColor White


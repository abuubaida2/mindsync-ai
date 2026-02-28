# MindSync — Start All Services
# Run this script from E:\app to launch both the API and Metro bundler

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Starting MindSync API (FastAPI)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', `
    "Set-Location '$root'; python -m uvicorn api.main:app --host 0.0.0.0 --port 8000" `
    -WindowStyle Normal

Start-Sleep 3

Write-Host "Starting Expo Metro Bundler..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList '-NoExit', '-Command', `
    "Set-Location '$root\mobile'; node node_modules\expo\bin\cli start --lan --port 8082" `
    -WindowStyle Normal

Write-Host ""
Write-Host "Both services launched in separate windows." -ForegroundColor Green
Write-Host "  API  -> http://localhost:8000" -ForegroundColor Yellow
Write-Host "  Expo -> exp://<your-ip>:8082  (scan QR in Expo Go)" -ForegroundColor Yellow

# MindSync Backend Startup Script
# Run this every time you want the app to work for external users
# Double-click this file OR run: powershell -ExecutionPolicy Bypass -File start-backend.ps1

Write-Host "Starting MindSync Backend..." -ForegroundColor Cyan

# Kill any existing backend/ngrok processes
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start FastAPI backend
Write-Host "Starting FastAPI backend on port 8000..." -ForegroundColor Yellow
Start-Process -FilePath "C:\Users\ABU UBAIDA\AppData\Local\Programs\Python\Python311\python.exe" `
    -ArgumentList "-m uvicorn api.main:app --host 0.0.0.0 --port 8000" `
    -WorkingDirectory "E:\app-mobile\backend" `
    -WindowStyle Normal

# Wait for backend to load models (~30 seconds)
Write-Host "Waiting for models to load (30 seconds)..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
        if ($r.status -eq "ok") { $ready = $true; break }
    } catch {}
    Write-Host "  Still loading... $([int](($i+1)*5))s" -ForegroundColor Gray
}

if (-not $ready) {
    Write-Host "Backend not ready after 120s - check the backend window for errors." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Backend ready!" -ForegroundColor Green

# Start ngrok with static domain
Write-Host "Starting ngrok tunnel..." -ForegroundColor Yellow
Start-Process -FilePath "ngrok" `
    -ArgumentList "http 8000" `
    -WindowStyle Normal

Start-Sleep -Seconds 5

# Verify tunnel
try {
    $tunnel = (Invoke-RestMethod "http://localhost:4040/api/tunnels").tunnels | Where-Object { $_.proto -eq "https" }
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host " MindSync is LIVE!" -ForegroundColor Green
    Write-Host " API URL: $($tunnel.public_url)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "The APK connects to:" -ForegroundColor White
    Write-Host "  https://unrealistic-lailah-godlessly.ngrok-free.dev" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Keep this window open while demos are running." -ForegroundColor Yellow
} catch {
    Write-Host "ngrok tunnel check failed - but backend is running on localhost:8000" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press Enter to keep running (DO NOT close this window)..." -ForegroundColor Yellow
Read-Host

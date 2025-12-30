# ETH Empirical Routes 5Y Backtest - PowerShell Runner
# Usage: .\run_all.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ETH Empirical Routes 5Y Backtest" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check Python
Write-Host "`nChecking Python installation..." -ForegroundColor Yellow
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "Python found: $($python.Source)" -ForegroundColor Green

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt --quiet

# Run backtest
Write-Host "`nStarting backtest..." -ForegroundColor Yellow
$startTime = Get-Date

python run_backtest.py --n-trials 100

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Backtest Complete!" -ForegroundColor Green
Write-Host "Duration: $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
Write-Host "Results: $scriptPath\outputs\" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Open outputs folder
if (Test-Path "$scriptPath\outputs") {
    Write-Host "`nOpening outputs folder..." -ForegroundColor Yellow
    explorer.exe "$scriptPath\outputs"
}


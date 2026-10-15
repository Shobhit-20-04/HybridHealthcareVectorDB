# ==============================================================================
# Helper Script: Install pgvector into PostgreSQL 18 (Requires Administrator)
# Project: Hybrid Relational–Vector Database Architecture for Healthcare
# ==============================================================================

# Check for Administrator elevation
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[!] This script requires Administrator privileges to copy extension files into 'C:\Program Files\PostgreSQL\18'." -ForegroundColor Yellow
    Write-Host "[*] Relaunching with Administrator elevation..." -ForegroundColor Cyan
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`""
    exit
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$zipPath = Join-Path $projectRoot "data\raw\vector.v0.8.6-pg18.zip"
$tempExtract = Join-Path $projectRoot "data\raw\vector_extracted"
$pgDir = "C:\Program Files\PostgreSQL\18"

Write-Host "==============================================================================" -ForegroundColor Green
Write-Host "  INSTALLING PGVECTOR v0.8.6 INTO POSTGRESQL 18" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green

if (-not (Test-Path $zipPath)) {
    Write-Error "[!] Archive not found: $zipPath"
    exit 1
}

Write-Host "[*] Extracting extension package..." -ForegroundColor Cyan
if (Test-Path $tempExtract) { Remove-Item -Path $tempExtract -Recurse -Force }
Expand-Archive -Path $zipPath -DestinationPath $tempExtract -Force

# 1. Copy DLL
$dllSource = Join-Path $tempExtract "lib\vector.dll"
$dllDest = Join-Path $pgDir "lib"
Write-Host "[*] Installing vector.dll -> $dllDest" -ForegroundColor Cyan
Copy-Item -Path $dllSource -Destination $dllDest -Force

# 2. Copy extension files
$extSource = Join-Path $tempExtract "share\extension\*"
$extDest = Join-Path $pgDir "share\extension"
Write-Host "[*] Installing SQL control scripts -> $extDest" -ForegroundColor Cyan
Copy-Item -Path $extSource -Destination $extDest -Force -Recurse

# 3. Clean temporary extraction
Remove-Item -Path $tempExtract -Recurse -Force

# 4. Restart PostgreSQL Windows Service
Write-Host "[*] Restarting postgresql-x64-18 service..." -ForegroundColor Cyan
Restart-Service -Name "postgresql-x64-18"

# 5. Verify installation files
$dllOk = Test-Path "$pgDir\lib\vector.dll"
$ctrlOk = Test-Path "$pgDir\share\extension\vector.control"

if ($dllOk -and $ctrlOk) {
    Write-Host "`n[V] SUCCESS: pgvector v0.8.6 successfully installed and service restarted!" -ForegroundColor Green
    Write-Host "    - vector.dll      : $pgDir\lib\vector.dll" -ForegroundColor Green
    Write-Host "    - vector.control  : $pgDir\share\extension\vector.control" -ForegroundColor Green
} else {
    Write-Host "`n[!] Installation verification failed. Please check folder permissions." -ForegroundColor Red
}

Write-Host "`nYou may now close this window and proceed." -ForegroundColor Yellow

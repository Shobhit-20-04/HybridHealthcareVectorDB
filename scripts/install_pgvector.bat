@echo off
:: ==============================================================================
:: One-Click pgvector Installer Launcher
:: Project: Hybrid Relational–Vector Database Architecture for Healthcare
:: ==============================================================================
echo Requesting Administrator privileges to copy pgvector files into PostgreSQL 18...
powershell -Command "Start-Process powershell.exe -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File """%~dp0install_pgvector_admin.ps1"""'"
pause

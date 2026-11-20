@echo off
chcp 65001 >nul
title web2kg Control Console
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_console.ps1" %*
if errorlevel 1 pause
